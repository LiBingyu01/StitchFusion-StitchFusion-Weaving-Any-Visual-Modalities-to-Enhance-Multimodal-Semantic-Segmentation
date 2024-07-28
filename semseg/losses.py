import torch
from torch import nn, Tensor
from torch.nn import functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, aux_weights: list = [1, 0.4, 0.4]) -> None:
        super().__init__()
        self.aux_weights = aux_weights
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        return self.criterion(preds, labels)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label: int = 255, weight: Tensor = None, thresh: float = 0.7, aux_weights: list = [1, 1]) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.aux_weights = aux_weights
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='none')

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        n_min = labels[labels != self.ignore_label].numel() // 16
        loss = self.criterion(preds, labels).view(-1)
        loss_hard = loss[loss > self.thresh]

        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)

        return torch.mean(loss_hard)

    def forward(self, preds, labels: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, labels) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, labels)


class Dice(nn.Module):
    def __init__(self, delta: float = 0.5, aux_weights: list = [1, 0.4, 0.4]):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.delta = delta
        self.aux_weights = aux_weights

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        num_classes = preds.shape[1]
        labels = F.one_hot(labels, num_classes).permute(0, 3, 1, 2)
        tp = torch.sum(labels*preds, dim=(2, 3))
        fn = torch.sum(labels*(1-preds), dim=(2, 3))
        fp = torch.sum((1-labels)*preds, dim=(2, 3))

        dice_score = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)
        dice_score = torch.sum(1 - dice_score, dim=-1)

        dice_score = dice_score / num_classes
        return dice_score.mean()

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


__all__ = ['CrossEntropy', 'OhemCrossEntropy', 'Dice', 'SupervisedMultiModalContrastiveLoss']


def get_loss(loss_fn_name: str = 'CrossEntropy', ignore_label: int = 255, cls_weights: Tensor = None):
    assert loss_fn_name in __all__, f"Unavailable loss function name >> {loss_fn_name}.\nAvailable loss functions: {__all__}"
    if loss_fn_name == 'Dice':
        return Dice()
    return eval(loss_fn_name)(ignore_label, cls_weights)



# 定义一个函数，用于检查张量是否包含无穷大或NaN值
def has_inf_or_nan(x):
    # 判断张量是否含有无穷大值或NaN值，并返回布尔值
    return torch.isinf(x).max().item(), torch.isnan(x).max().item()


# 定义一个监督多模态对比损失类，该类继承自nn.Module
class SupervisedMultiModalContrastiveLoss(nn.Module):
    # 类的初始化函数，参数config包含了训练配置信息
    def __init__(self, num_all_classes, config):
        super().__init__()  # 调用父类的初始化方法

        # 从配置信息中提取相关参数，并设置为类的成员变量
        self.num_all_classes = num_all_classes
        self.min_views_per_class = config["min_views_per_class"]
        self.max_views_per_class = config["max_views_per_class"]
        self.max_features_total = config["max_features_total"]

        self.cross_modal_temperature = config["cm_temperature"]
        self.temperature = config["temperature"]

        self._scale = None  # 初始化一个内部变量_scale

    # 定义类的前向传播函数，输入参数包括标签和两种模态的特征
    def forward(self, label: torch.Tensor, color_fea: torch.Tensor, other_fea: torch.Tensor):
        # 使用torch.no_grad()来避免计算梯度，提高计算效率
        with torch.no_grad():
            # 计算缩放比例，用于后续处理
            # scale_h = label.shape[-2] // color_fea.shape[-2]
            # scale_w = label.shape[-1] // color_fea.shape[-1]
            # 获取每个尺度下的类别分布和主导类别
            # class_distribution：每个像素点分类
            _, class_distribution = self.get_dist_and_classes(label, color_fea)
            # print("label",label.shape)
            # print("other_fea",other_fea.shape)
            # print("class_distribution",class_distribution.shape)
            # 根据每个像素的分类采样每个类别的像素，和该像素对应的类别
            color_feas, labels = self.sample_anchors_fast(class_distribution, color_fea)
            other_feas, _ = self.sample_anchors_fast(class_distribution, other_fea)

            # 计算多模态对比损失
            loss_cm = self.cm_contrastive_loss(color_feas, other_feas, labels)
            # 计算视觉模态对比损失
            loss_vis = self.contrastive_loss(color_feas, labels)
            # 计算辅助模态对比损失
            loss_aux = self.contrastive_loss(other_feas, labels)
            # 返回计算的所有损失
            # print(loss_cm)
            # print(loss_vis)
            # print(loss_aux)
            return loss_cm, loss_vis, loss_aux

    # 该函数确定每个scale*scale尺度的标签图块中的类分布情况，并返回相应尺度的类分布tensor。
    def get_dist_and_classes(self, label: torch.Tensor, color_fea) -> torch.Tensor:
        """
        确定每个scale*scale大小的地面真实标签块N-H-W中的类分布情况，
        返回的类分布形状为N-C-H//scale-W//scale的tensor。

        同时基于类分布确定每个地面真实标签块N-H-W中的【主导类】。

        输出是N-C-H//scale-W//scale，其中C可能是1（只有一个主导类）或更多。
        如果label_scaling_mode == 'nn'，则对标签执行最近邻插值，不进行独热编码，
        并返回N-1-H//scale-W//scale的tensor。
        """
        n, h, w = label.shape  # 获取标签的形状
        # 使用[最近邻插值将标签下采样到相应的尺度]
        label_down = torch.nn.functional.interpolate(label.unsqueeze(1).float(), (color_fea.shape[-2], color_fea.shape[-1]),
                                                     mode='nearest')
        return label_down.long(), label_down.long()  # 返回下采样后的标签

    # 快速采样锚点的函数
    def sample_anchors_fast(self, dominant_classes, feature):
        """
        input: dominant_classes N-1-H-W
               features N-C-H-W
        返回采样的特征和标签：
               sampled_features T-C-V
               sampled_labels T
               T: 批次中的类/锚点（可能会重复）
               C: 特征空间的维度
               V: 每个类/锚点的视图数，即每个类/锚点的样本数
        """
        n = dominant_classes.shape[0] # 批量大小
        c = feature.shape[1]  # 特征维度
        feature = feature.view(n, c, -1)  # 将特征展平 【16,3,64,64】
        dominant_classes = dominant_classes.view(n, -1)  # 将主导类展平 【16,1,64,64】

        # 创建类的索引
        classes_ids = torch.arange(start=0, end=self.num_all_classes, step=1, device=dominant_classes.device)
        # 判断dominant_classes中的类是否与类索引匹配
        compare = dominant_classes.unsqueeze(-1) == classes_ids.unsqueeze(0).unsqueeze(0) # 每一个像素的分类 onehot
        # 计算每个类在批次中出现的次数
        cls_counts = compare.sum(1)

        # 找出满足最小【像素】数要求的类的索引
        present_ids = torch.where(cls_counts >= self.min_views_per_class)
        
        # batch_ids：类别在第几个图片中tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4,
        # 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
        # 7, 7, 7, 7, 7, 7, 7, 7, 7], device='cuda:0')

        # cls_in_batch：每张图片的类别标签进行了展开tensor([ 5,  6,  8, 13,  3,  5,  6, 13,  0,  1,  2,  5,  6,  7,  8, 13,  2,  3,
        #  4,  5,  6,  8, 13,  0,  1,  5,  6,  7,  8, 11,  0,  1,  2,  3,  5,  6,
        #  7,  8, 12, 13,  0,  1,  2,  5,  6,  7,  8, 13,  0,  1,  2,  3,  5,  6,
        #  7,  8, 13], device='cuda:0')

        batch_ids, cls_in_batch = present_ids
        # 获取每个类的最小像素数
        min_views = torch.min(cls_counts[present_ids])
        total_cls = cls_in_batch.shape[0]  # 总共的类数
        cls_counts_in_batch = cls_counts
        # 选择每个类的视图数-【像素数】
        views_per_class = self._select_views_per_class(min_views, total_cls, cls_in_batch, cls_counts_in_batch)
        # 根据每个类别的最小视图数、总类别数、批次中的类别和类别的计数来确定每个类别要采样的视图数
        # 初始化采样特征和标签的tensor，不在GPU上
        # (total_cls, c, views_per_class)：
        sampled_features = torch.zeros((total_cls, c, views_per_class), dtype=torch.float, device=batch_ids.device)
        sampled_labels = torch.zeros(total_cls, dtype=torch.float, device=batch_ids.device)
        # 为每个类别创建特征和标签的张量，其中特征张量的形状是(总类别数, 特征维度, 每类视图数)，标签张量的长度是总类别数
        # 对每个【类】进行采样
        for i in range(total_cls):
            # 获得每一张图片中类别为cls_in_batch[i]的所有像素
            # 对于batch中的每一个图片，用cls_in_batch[i]索引出我应该采样的哪几类【dominant】的像素,
            # 之后nonzero去掉像素图中不是这个类别的部分，也就是只保存nonzero的部分
            # indices_from_cl是一个向量，向量中是这个图片batch_ids[i]，这一类cls_in_batch[i]的像素编号
            indices_from_cl = compare[batch_ids[i], :, cls_in_batch[i]].nonzero().squeeze() # 
             # 对索引进行随机排列以后续随机抽取
            random_permutation = torch.randperm(indices_from_cl.shape[0])
            # 获取前views_per_class个
            sampled_indices = random_permutation[:views_per_class] # random_permutation[:views_per_class]采样的是前最小采样像素数的随机采样数，
            # 从类别标签中采样
            cl = indices_from_cl[sampled_indices]
            # -------------- 实现采样操作，从像素中获取第batch_ids[i]个图片的cl为编号的像素点
            sampled_features[i] = feature[batch_ids[i], :, cl]
            # 设置相应的标签
            sampled_labels[i] = cls_in_batch[i]

        # 返回采样后的特征和标签张量
        return sampled_features, sampled_labels

    # 此函数用于选择每个类别要采样的视图数，以平衡类别间的样本数并避免内存溢出。
    def _select_views_per_class(self, min_views, total_cls, cls_in_batch, cls_counts_in_batch):
        
        if self.max_views_per_class == 1:
            # 如果每个类别的最大视图数为1，则不限制视图数
            views_per_class = min_views
        else:
            # 否则，为了避免内存溢出，将视图数限制在一个最大值
            views_per_class = min(min_views, self.max_views_per_class)

        # 如果计算出的视图数乘以总类别数超过了设定的最大特征数，需要进一步减少视图数
        if views_per_class * total_cls > self.max_features_total:
            views_per_class = self.max_features_total // total_cls

        return views_per_class  # 返回确定的每个类别的视图数

    # 跨模态对比学习
    def cm_contrastive_loss(self, feats_c, feats_o, labels):
        """
        输入:
            feats T-C-V
                T: 批次中的类别/锚点（可能重复）
                C: 特征空间的维度
                V: 每个类别/锚点的视图数，即每个类别/锚点的样本数
            labels T: 标签
        返回:
            loss: 损失值
        """
        # 准备颜色特征，进行L2归一化
        feats_c = torch.nn.functional.normalize(feats_c, p=2, dim=1)
        # 转置特征，方便后续处理
        feats_c = feats_c.transpose(dim0=1, dim1=2)
        # 获取锚点数，每个锚点的视图数，和特征的维度
        num_anchors, views_per_anchor, c = feats_c.shape
        # 将归一化的特征展平
        feats_c_flat = feats_c.contiguous().view(-1, c)
        # 准备其他模态的特征，进行L2归一化
        feats_o = torch.nn.functional.normalize(feats_o, p=2, dim=1)
        # 同样转置特征
        feats_o = feats_o.transpose(dim0=1, dim1=2)
        # 确认锚点数，每个锚点的视图数，和特征的维度
        num_anchors, views_per_anchor, c = feats_o.shape
        # 展平其他模态的特征
        feats_o_flat = feats_o.contiguous().view(-1, c)
        # 将标签转换为连续的张量，并调整形状以适配视图数
        labels = labels.contiguous().view(-1, 1)
        labels = labels.repeat(1, views_per_anchor)
        labels = labels.view(-1, 1)

        # 获取正样本和负样本的掩码
        pos_mask, neg_mask = self.get_masks(labels, num_anchors, views_per_anchor)

        # 计算颜色特征和其他模态特征之间的点乘，并用交叉模态温度参数进行缩放
        dot_product = torch.div(torch.matmul(feats_c_flat, torch.transpose(feats_o_flat, 0, 1)),
                                self.cross_modal_temperature)
        # 计算InfoNCE损失
        loss = self.InfoNCE_loss(pos_mask, neg_mask, dot_product)
        return loss  # 返回计算得到的多模态对比损失

    # 对比损失的计算
    def contrastive_loss(self, feats, labels):
        """
        输入：feats T-C-V
            T: BATCH中的类/锚点数（可能重复）
            C: 特征空间的维度
            V: 每个类/锚点的视图数，即每个类/锚点的样本数
              labels T
        返回：loss 损失值
        """
        # 准备特征，进行归一化处理
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        # 转置以便后续计算
        feats = feats.transpose(dim0=1, dim1=2)
        # 获取锚点数、每个锚点的视图数和特征维数
        num_anchors, views_per_anchor, c = feats.shape

        # 整理标签以用于后续计算
        labels = labels.contiguous().view(-1, 1)
        labels_ = labels.repeat(1, views_per_anchor)
        labels_ = labels_.view(-1, 1)

        # 获取正负样本掩码
        pos_mask, neg_mask = self.get_masks(labels_, num_anchors, views_per_anchor)
        # 将特征展平以用于矩阵乘法
        feats_flat = feats.contiguous().view(-1, c)
        # 计算特征间的点乘，并根据温度参数进行缩放【相似性矩阵】
        dot_product = torch.div(torch.matmul(feats_flat, torch.transpose(feats_flat, 0, 1)), self.temperature)
        # 计算InfoNCE损失
        loss = self.InfoNCE_loss(pos_mask, neg_mask, dot_product)
        return loss  # 返回损失值

    # 静态方法，用于获取正负样本的掩码
    @staticmethod
    def get_masks(labels, num_anchors, views_per_anchor):
        """
        输入扁平化的标签，确定每个锚点的正负样本
        参数：labels: T*V-1
              num_anchors: T
              views_per_anchor: V
        返回：pos_mask, neg_mask 正样本掩码和负样本掩码
        """ 
        # 提取指示同类样本的掩码
        mask = torch.eq(labels, torch.transpose(labels, 0, 1)).float()
        neg_mask = 1 - mask  # 指示负样本的掩码

        # 将对角线掩码元素设置为零，排除自身比较的情况
        logits_mask = torch.ones_like(mask).scatter_(1,torch.arange(num_anchors * views_per_anchor,device=mask.device).view(-1, 1),0)
        pos_mask = mask * logits_mask  # 指示正样本的掩码
        return pos_mask, neg_mask  # 返回正负样本掩码

    # InfoNCE损失的计算
    def InfoNCE_loss(self, pos, neg, dot):
        """
        参数：pos: V*T-V*T 正样本矩阵
              neg: V*T-V*T 负样本矩阵
              dot: V*T-V*T 点乘结果矩阵
        返回：loss 损失值
        """
        # 计算logits，这里直接使用dot，不进行最大值归一化
        logits = dot  # - logits_max.detach()

        # 计算负样本的指数和
        neg_logits = torch.exp(logits) * neg
        neg_logits = neg_logits.sum(1, keepdim=True)

        # 计算所有样本的指数
        exp_logits = torch.exp(logits)

        # 计算log概率
        log_prob = logits - torch.log(exp_logits + neg_logits)

        # 根据正样本的数量进行规范化
        pos_sums = pos.sum(1)
        ones = torch.ones(size=pos_sums.size())
        norm = torch.where(pos_sums > 0, pos_sums, ones.to(pos.device))
        mean_log_prob_pos = (pos * log_prob).sum(1) / norm

        # 计算最终的损失
        loss = - mean_log_prob_pos

        loss = loss.mean()  # 计算损失的均值
        # 检查损失中是否包含无穷大或NaN
        if has_inf_or_nan(loss)[0] or has_inf_or_nan(loss)[1]:
            print('\n inf found in loss with Positives {} and Negatives {}'.format(pos.sum(1), neg.sum(1)))

        return loss  # 返回损失值


if __name__ == '__main__':
    pred = torch.randint(0, 19, (2, 19, 480, 640), dtype=torch.float)
    label = torch.randint(0, 19, (2, 480, 640), dtype=torch.long)
    loss_fn = Dice()
    y = loss_fn(pred, label)
    print(y)