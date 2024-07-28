import torch
import argparse
import yaml
import math
import os
import time
import sys
module_dir = "Path/to/stitchfusion/"
if module_dir not in sys.path:
    sys.path.append(module_dir)
# 设置OPENBLAS_NUM_THREADS为64
os.environ["OPENBLAS_NUM_THREADS"] = "64"
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations_mm import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from math import ceil
import numpy as np
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp, get_logger, cal_flops, print_iou
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.utils import resample
def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    padded_img = F.pad(img, (0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

@torch.no_grad()
def sliding_predict(model, image, num_classes, flip=True):
    image_size = image[0].shape
    tile_size = (int(ceil(image_size[2]*1)), int(ceil(image_size[3]*1)))
    overlap = 1/3

    stride = ceil(tile_size[0] * (1 - overlap))
    
    num_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)
    num_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)
    total_predictions = torch.zeros((num_classes, image_size[2], image_size[3]), device=torch.device('cuda'))
    count_predictions = torch.zeros((image_size[2], image_size[3]), device=torch.device('cuda'))
    tile_counter = 0

    for row in range(num_rows):
        for col in range(num_cols):
            x_min, y_min = int(col * stride), int(row * stride)
            x_max = min(x_min + tile_size[1], image_size[3])
            y_max = min(y_min + tile_size[0], image_size[2])

            img = [modal[:, :, y_min:y_max, x_min:x_max] for modal in image]
            padded_img = [pad_image(modal, tile_size) for modal in img]
            tile_counter += 1
            padded_prediction,_,_ = model(padded_img)
            if flip:
                fliped_img = [padded_modal.flip(-1) for padded_modal in padded_img]
                fliped_predictions,_,_ = model(fliped_img)
                padded_prediction += fliped_predictions.flip(-1)
            predictions = padded_prediction[:, :, :img[0].shape[2], :img[0].shape[3]]
            count_predictions[y_min:y_max, x_min:x_max] += 1
            total_predictions[:, y_min:y_max, x_min:x_max] += predictions.squeeze(0)

    return total_predictions.unsqueeze(0)

@torch.no_grad()
def evaluate(model, dataloader, device, VIS_Saving, loss_fn=None):
    print('Evaluating...')
    model.eval()
    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)
    sliding = False
    test_loss = 0.0
    iter = 0
    i = 0

    np.random.seed(42)  # 你可以使用任何数字作为种子
    for images, labels in tqdm(dataloader):
        images = [x.to(device) for x in images]
        labels = labels.to(device)
        
        if sliding:
            # preds = sliding_predict(model, images, num_classes=n_classes).softmax(dim=1)
            preds,_,_ = sliding_predict(model, images, num_classes=n_classes)
        else:
            # preds = model(images).softmax(dim=1)
            preds = model(images)
        
        # 为每个类别定义一个颜色
        colors = [
            (0.2745098, 0.2745098, 0.2745098),  # 灰色
            (0.3921569, 0.1568627, 0.1568627),  # 红色
            (0.2156863, 0.3568628, 0.3137255),  # 绿色
            (0.7843137, 0.0784314, 0.2352941),  # 橙色
            (0.6039216, 0.6039216, 0.6039216),  # 灰色
            (0.6196079, 0.9215686, 0.1960784),  # 黄色
            (0.4980392, 0.2509804, 0.4980392),  # 紫色
            (0.9647059, 0.1372549, 0.9019608),  # 粉色
            (0.4196078, 0.5568627, 0.1372549),  # 绿色
            (0, 0, 0.5568627),  # 蓝色
            (0.4, 0.4, 0.6156863),  # 灰色
            (0.8588235, 0.8588235, 0),  # 黄色
            (0.2745098, 0.509804, 0.7058824),  # 蓝色
            (0.3176471, 0, 0.3176471),  # 紫色
            (0.5882353, 0.3921569, 0.3921569),  # 灰色
            (0.9019608, 0.5882353, 0.5450981),  # 橙色
            (0.7058824, 0.6470588, 0.7058824),  # 灰色
            (0.9803922, 0.6666667, 0.1176471),  # 黄色
            (0.4313726, 0.7450981, 0.6313726),  # 蓝色
            (0.6588235, 0.4705882, 0.1960784),  # 棕色
            (0.1764706, 0.2352941, 0.5882353),  # 蓝色
            (0.5647059, 0.6666667, 0.3921569),  # 绿色
            (0, 0, 0.9019608),  # 蓝色
            (0, 0.2352941, 0.3921569),  # 绿色
            (0, 0, 0.2745098)  # 蓝色
        ]

        labels_view = labels.view(labels.shape[0],-1).cpu().numpy().squeeze()
        # --------------pred
        preds = preds.squeeze(0)
        preds_view = preds.view(preds.shape[0],-1).transpose(1,0)
        preds_view = preds_view[labels_view != 255,:]
        # --------------label
        labels_view = labels_view[labels_view != 255]
        # --------------采样
        unique_labels, counts = np.unique(labels_view, return_counts=True)
        
        # 找出数量最少的类别
        min_size_1 = counts.min() # label的最小数
        if min_size_1 < 500:
            continue
        else: 
            min_size = 500
        
        # 对每个类别进行相同数量的数据采样
        resampled_features = []
        resampled_labels = []

        for label in unique_labels:
            np.random.seed(0)
            # 筛选当前类别的特征和标签
            class_features = preds_view[labels_view == label]
            class_labels = labels_view[labels_view == label]
            # 对数据进行重采样，确保每个类别的数量相同
            resampled_class_features, resampled_class_labels = resample(
                class_features, class_labels, 
                replace=False, 
                n_samples=min_size, 
                random_state=0
            )
            
            # 将重采样后的数据添加到列表中
            resampled_features.append(resampled_class_features)
            resampled_labels.append(resampled_class_labels)

        # 合并所有重采样后的数据
        resampled_features = [tensor.cpu() for tensor in resampled_features]
        resampled_features = np.vstack(resampled_features)
        resampled_labels = np.concatenate(resampled_labels)
        print("重采样后的特征形状:", resampled_features.shape)
        print("重采样后的标签形状:", resampled_labels.shape)
        print("总共有", len(unique_labels), "个类别")

        preds_view_sampled = resampled_features
        labels_view_sample = resampled_labels
        # --------------figure
        image_path = f'{VIS_Saving}/tsne_visualization_{i}.png'
        cmap = ListedColormap(colors)
        plt.figure(figsize=(8, 8))
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(preds_view_sampled)  # 重塑为2D矩阵以符合fit_transform的输入
        # 可视化t-SNE结果 
        plt.figure(figsize=(6, 6))
        colors_fig = [colors[labels_view_sample[idx]] for idx in range(len(labels_view_sample))]
        preds_view_sampled
        plt.scatter(features_2d[:, 0], features_2d[:, 1], c=colors_fig, marker='o')
        plt.title('t-SNE Visualization of Image Features')
        plt.xlabel('t-SNE feature 0')
        plt.ylabel('t-SNE feature 1')
        plt.savefig(image_path,dpi=1000)
        i += 1
    return 0


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = [F.interpolate(img, size=(new_H, new_W), mode='bilinear', align_corners=True) for img in images]
            scaled_images = [scaled_img.to(device) for scaled_img in scaled_images]
            logits,_,_ = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = [torch.flip(scaled_img, dims=(3,)) for scaled_img in scaled_images]
                logits,_,_ = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    eval_cfg = cfg['EVAL']
    VIS_Saving = eval_cfg['VIS_TSNE_SAVE_DIR']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    # cases = ['cloud', 'fog', 'night', 'rain', 'sun']
    # cases = ['motionblur', 'overexposure', 'underexposure', 'lidarjitter', 'eventlowres']
    cases = [None] # all
    
    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): 
        raise FileNotFoundError
    print(f"Evaluating {model_path}...")

    exp_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    eval_path = os.path.join(os.path.dirname(eval_cfg['MODEL_PATH']), 'eval_{}.txt'.format(exp_time))

    for case in cases:
        dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'val', transform, cfg['DATASET']['MODALS'], case)
        
        # --- test set
        # dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'test', transform, cfg['DATASET']['MODALS'], case)

        model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes, cfg['DATASET']['MODALS'])
        
        msg = model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        print(msg)
        model = model.to(device)
        sampler_val = None
        dataloader = DataLoader(dataset, batch_size=eval_cfg['BATCH_SIZE_VIS'], num_workers=eval_cfg['BATCH_SIZE'], pin_memory=False, sampler=sampler_val)
        
        ok = evaluate(model, dataloader, device,VIS_Saving)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='Path/to/stitchfusion/configs/deliver.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    # gpu = setup_ddp()
    # main(cfg, gpu)
    main(cfg)