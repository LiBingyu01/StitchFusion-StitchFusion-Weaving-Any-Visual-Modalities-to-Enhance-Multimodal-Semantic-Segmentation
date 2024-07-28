import os
import sys
module_dir = "Path/to/stitchfusion"
if module_dir not in sys.path:
    sys.path.append(module_dir)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import torch
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
from tools_no_contrast.val_mm import evaluate
from semseg.metrics import Metrics
import gc
def main(cfg, save_dir):
# ------------------------------ 提前准备和编辑 ---------------------------------------
    
    start = time.time()  # 记录训练开始的时间
    best_mIoU = 0.0  # 初始化最佳mean Intersection over Union（mIoU）为0
    best_epoch = 0  # 记录获得最佳mIoU的epoch
    num_workers = 2  # DataLoader的工作进程数
    device = torch.device(cfg['DEVICE'])  # 设置设备，如'cuda'或'cpu'
    
    # ------ 从配置中提取相关配置子字典
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']  # 训练周期和学习率
    resume_path = cfg['MODEL']['RESUME']  # 恢复训练的检查点路径
    gpus = cfg['GPUs']  # GPU设备列表
    use_wandb = cfg['USE_WANDB']  # 是否使用Weights & Biases进行实验跟踪
    # ------ 获取数据增强方法
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])
    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    
    # ------ 实例化训练和验证数据集
    print(dataset_cfg['NAME'])
    print(dataset_cfg['ROOT'])
    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform, dataset_cfg['MODALS'])
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform, dataset_cfg['MODALS'])
    
    class_names = trainset.CLASSES  # 类别名列表
    num_all_classes = len(class_names)
    
    # ------ 实例化模型
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes, dataset_cfg['MODALS'])
    
    # ------ 预训练模型加载
    # 如果有可用的检查点文件，则从该检查点恢复模型
    resume_checkpoint = None
    if os.path.isfile(resume_path):
        resume_checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
        adjusted_state_dict = {k.replace('module.', ''): v for k, v in resume_checkpoint['model_state_dict'].items()}
        msg = model.load_state_dict(adjusted_state_dict)
        # msg = model.load_state_dict(resume_checkpoint['model_state_dict'])
        # print(msg)
        logger.info(msg)
    else:# 否则，从预训练模型初始化
        model.init_pretrained(model_cfg['PRETRAINED']) # 记载预训练模型

    # 使用DataParallel进行模型并行化，可在多个GPU上运行模型
    model = torch.nn.DataParallel(model, device_ids=cfg['GPU_IDs'])
    # 计算模型的总参数数（仅包括可训练参数）
    model = model.to(device)
    
    # 计算每个epoch的迭代次数
    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # 获取损失函数
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)

    # 开始的epoch默认为0
    start_epoch = 0
    # 初始化优化器
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    # 初始化调度器（用于调整学习率）
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, int((epochs+1)*iters_per_epoch), sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'])

    # 如果使用分布式数据并行（DDP）
    if train_cfg['DDP']: 
        # 初始化分布式采样器，确保每个进程获取到不同部分的数据
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        # 验证集的采样器设置为None
        sampler_val = None
        # 将模型包装成DDP模型
        model = DDP(model, device_ids=[gpus], output_device=0, find_unused_parameters=True)
    else:
        # 否则，使用随机采样器
        sampler = RandomSampler(trainset)
        # 验证集的采样器设置为None
        sampler_val = None
    
    # 如果有检查点可用，从检查点恢复训练
    if resume_checkpoint:
        # 设置开始的epoch
        start_epoch = resume_checkpoint['epoch'] - 1
        # 从检查点恢复优化器的状态
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        # 从检查点恢复调度器的状态
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
        # 从检查点恢复损失
        loss = resume_checkpoint['loss']        
        # 从检查点恢复最佳mIoU
        best_mIoU = resume_checkpoint['best_miou']
    # ------ 数据集加载器       
    # 初始化训练数据加载器    
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=False, sampler=sampler)
    # 初始化验证数据加载器
    valloader = DataLoader(valset, batch_size=eval_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=False, sampler=sampler_val)
    # 统计训练配置和参数数目，并打印在tensorboard
    scaler = GradScaler(enabled=train_cfg['AMP'])
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        writer = SummaryWriter(str(save_dir))
        logger.info('================== model complexity =====================')
        cal_flops(model, dataset_cfg['MODALS'], logger)
        logger.info('================== model structure =====================')
        logger.info(model)
        logger.info('================== training config =====================')
        logger.info(cfg)
        logger.info('================== parameter count =====================')
        logger.info(sum(p.numel() for p in model.parameters() if p.requires_grad))

# ------------------------------开始训练---------------------------------------
    for epoch in range(start_epoch, epochs):
        # Clean Memory
        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0  
        lr = scheduler.get_lr()# 获取当前学习率
        lr = sum(lr) / len(lr)# 计算平均学习率

        # 使用tqdm创建进度条，遍历trainloader数据，total设置为每个epoch的迭代次数
        # 设置进度条的描述信息，包括当前epoch、迭代次数、学习率和损失值
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        metrics = Metrics(trainset.n_classes, trainloader.dataset.ignore_label, device)
   
        # 开始遍历每个batch进行训练
        for iter, (sample, lbl) in pbar:
            optimizer.zero_grad(set_to_none=True)
            sample = [x.to(device) for x in sample]
            # print(sample)
            lbl = lbl.to(device)
            with autocast(enabled=train_cfg['AMP']):
                logits = model(sample)
                # 交叉熵
                loss = loss_fn(logits, lbl)
                loss_add = loss
            
            metrics.update(logits.softmax(dim=1), lbl)  # 更新度量状态
            
            scaler.scale(loss_add).backward()  # 反向传播，计算梯度
            scaler.step(optimizer)  # 利用梯度进行优化更新
            scaler.update()  # 更新梯度缩放器
            scheduler.step()  # 更新学习率调度器
            torch.cuda.synchronize()  # 同步所有CUDA操作
            
            lr = scheduler.get_lr()  # 获取当前学习率
            lr = sum(lr) / len(lr)  # 计算平均学习率
            if lr <= 1e-8: lr = 1e-8 # if lr is less than 1e-8, set it to 1e-8 (minimum lr)
            train_loss += loss_add.item()  # 累计单轮epoch的损失

            # Clean Memory
            torch.cuda.empty_cache()
            gc.collect()
            # 更新进度条描述信息，显示当前epoch、迭代次数、学习率和平均损失
            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} LOSS: {train_loss / (iter+1):.8f}")
            
        # 计算并更新整个epoch的平均训练损失
        train_loss /= iter+1
        # 如果是主进程（对于分布式训练）或不使用分布式训练，则记录训练损失到tensorboard
        if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
            writer.add_scalar('train/loss', train_loss, epoch)
        # 清空CUDA缓存，帮助减少显存占用
        torch.cuda.empty_cache()

        # 计算并获取IoU、平均IoU、像素准确率、平均像素准确率、F1分数和平均F1分数
        ious, miou = metrics.compute_iou()
        acc, macc = metrics.compute_pixel_acc()
        f1, mf1 = metrics.compute_f1()

        # 准备训练日志数据，包括epoch、训练损失、mIoU、像素准确率和F1分数
        train_log_data = {
            "Epoch": epoch+1,
            "Train Loss": train_loss,
            "Train mIoU": miou,
            "Train Pixel Acc": macc,
            "Train F1": mf1,
        }

# ============================== 评估 ===================================        
        # 每隔一定epoch或在最后一个epoch，进行模型评估
        if ((epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 and (epoch+1)>train_cfg['EVAL_START']) or (epoch+1) == epochs:
            if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
                acc, macc, f1, mf1, ious, miou, test_loss = evaluate(model, valloader, device, loss_fn=loss_fn)
                writer.add_scalar('val/mIoU', miou, epoch)
                # if use wandb
                log_data = {
                    "Test Loss": test_loss,
                    "Test mIoU": miou,
                    "Test Pixel Acc": macc,
                    "Test F1": mf1,
                }
                log_data.update(train_log_data)
                print(log_data)
                if use_wandb:
                    wandb.log(log_data)
                # 如果当前的 miou（Mean Intersection over Union）大于之前记录的最佳 miou，则执行以下操作：
                if miou > best_mIoU:
                    # 设置先前最佳模型的检查点文件路径
                    prev_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    # 设置先前最佳模型文件路径
                    prev_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    # 如果先前最佳模型文件存在，则删除
                    if os.path.isfile(prev_best):
                        os.remove(prev_best)
                    # 如果先前最佳模型的检查点文件存在，则删除
                    if os.path.isfile(prev_best_ckp):
                        os.remove(prev_best_ckp)
                    # 更新最佳 miou 和对应的 epoch
                    best_mIoU = miou
                    best_epoch = epoch + 1
                    # 设置当前最佳模型的检查点文件路径
                    cur_best_ckp = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}_checkpoint.pth"
                    # 设置当前最佳模型文件路径
                    cur_best = save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}_epoch{best_epoch}_{best_mIoU}.pth"
                    # 保存当前最佳模型
                    torch.save(model.module.state_dict(), cur_best)
                    # 保存模型的训练状态和相关信息到检查点文件
                    torch.save(
                        {
                            'epoch': best_epoch,
                            'model_state_dict': model.module.state_dict() if train_cfg['DDP'] else model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': train_loss,
                            'scheduler_state_dict': scheduler.state_dict(),
                            'best_miou': best_mIoU,
                        },
                        cur_best_ckp
                    )
                    # 记录并打印当前的 miou、acc、macc 和 class_names
                    logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))
                logger.info(f"Current epoch:{epoch} mIoU: {miou} Best mIoU: {best_mIoU}")        
    # 如果是分布式训练（DDP）且当前进程是主进程（rank == 0），或者不是分布式训练
    if (train_cfg['DDP'] and torch.distributed.get_rank() == 0) or (not train_cfg['DDP']):
        # 关闭 Tensorboard writer
        writer.close()
    # 关闭进度条
    pbar.close()
    # 计算总体训练时间，并格式化为时:分:秒
    end = time.gmtime(time.time() - start)
    # 构建用于打印的信息表格
    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    # 使用 tabulate 函数将信息表格转为字符串形式，并对齐数字在右侧
    logger.info(tabulate(table, numalign='right'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='Path/to/configs/mcubes_rgbadn.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    # gpu = setup_ddp()
    modals = ''.join([m[0] for m in cfg['DATASET']['MODALS']])
    model = cfg['MODEL']['BACKBONE']
    # exp_name = '_'.join([cfg['DATASET']['NAME'], model, modals])
    exp_name = cfg['WANDB_NAME']
    if cfg['USE_WANDB']:
        wandb.init(project="ProjcetName", entity="EntityName", name=exp_name)

    save_dir = Path(cfg['SAVE_DIR'], exp_name)
    if os.path.isfile(cfg['MODEL']['RESUME']):
        save_dir =  Path(os.path.dirname(cfg['MODEL']['RESUME']))
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / 'train.log')
    main(cfg, save_dir)
    cleanup_ddp()