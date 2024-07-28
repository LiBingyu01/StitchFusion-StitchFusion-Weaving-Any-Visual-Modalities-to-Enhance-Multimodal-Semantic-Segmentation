import torch
from torch import nn, Tensor
from torch.nn import functional as F
from semseg.models.layers import DropPath
import torch.nn.init as init
import numpy as np
from math import factorial
# ------------------------------------- fusion block 模块 -------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))

# 卷积核1
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

# 卷积核2    
class CustomDWConv(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

# 卷积核3
class CustomPWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)

        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.bn(self.pwconv(x))
        return x.flatten(2).transpose(1, 2)

class Bi_direct_adapter(nn.Module):
    def __init__(self, dim, xavier_init=False):
        super().__init__()

        self.adapter_down = nn.Linear(dim, 8)  
        self.adapter_mid = nn.Linear(8, 8)
        self.adapter_up = nn.Linear(8, dim)  

        #nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_mid.bias)
        nn.init.zeros_(self.adapter_mid.weight)
        nn.init.zeros_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)   
        #x_down = self.act(x_down)
        x_down = F.gelu(self.adapter_mid(x_down))
        #x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  
        #print("return adap x", x_up.size())
        return x_up #.permute(0, 2, 1).reshape(B, C, H, W)

# ------------------------------------- segformer 模块 -------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class PatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=32, patch_size=7, stride=4, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, patch_size, stride, padding)    # padding=(ps[0]//2, ps[1]//2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

# 通道注意力机制模块
class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

        # Initialize linear layers with Kaiming initialization
        for m in self.fc:
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, H, W):
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return (x * y.expand_as(x)).flatten(2).transpose(1, 2)

class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.pwconv1 = CustomPWConv(c2)
        self.dwconv3 = CustomDWConv(c2, 3)
        self.dwconv5 = CustomDWConv(c2, 5)
        self.dwconv7 = CustomDWConv(c2, 7)
        self.pwconv2 = CustomPWConv(c2)
        self.fc2 = nn.Linear(c2, c1)

        # Initialize fc1 layer with Kaiming initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        x = self.fc1(x)
        x = self.pwconv1(x, H, W)
        x1 = self.dwconv3(x, H, W)
        x2 = self.dwconv5(x, H, W)
        x3 = self.dwconv7(x, H, W)
        return self.fc2(F.gelu(self.pwconv2(x + x1 + x2 + x3, H, W)))


class FeatureCross(nn.Module):
    def __init__(self, channels, num_modals):
        super(FeatureCross, self).__init__()
        self.channels = channels
        self.num_modals = num_modals

        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])

        self.mix_ffn = nn.ModuleList([
            MixFFN(self.channels[0], self.channels[0]),
            MixFFN(self.channels[1], self.channels[1]),
            MixFFN(self.channels[2], self.channels[2]),
            MixFFN(self.channels[3], self.channels[3]),
        ])

        self.channel_attns = nn.ModuleList([
            ChannelAttentionBlock(self.channels[0]),
            ChannelAttentionBlock(self.channels[1]),
            ChannelAttentionBlock(self.channels[2]),
            ChannelAttentionBlock(self.channels[3]),
        ])

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        # conv fusion
        x = torch.cat(x, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_sum = self.liner_fusion_layers[layer_idx](x)
        # x_sum = self.mix_ffn[layer_idx](x_sum, H, W)
        x_sum = self.mix_ffn[layer_idx](x_sum, H, W)  + self.channel_attns[layer_idx](x_sum, H, W)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2)

class FeatureConc(nn.Module):
    def __init__(self, channels, num_modals):
        super(FeatureConc, self).__init__()
        self.channels = channels
        self.num_modals = num_modals

        self.liner_fusion_layers = nn.ModuleList([
            nn.Linear(self.channels[0]*self.num_modals, self.channels[0]),
            nn.Linear(self.channels[1]*self.num_modals, self.channels[1]),
            nn.Linear(self.channels[2]*self.num_modals, self.channels[2]),
            nn.Linear(self.channels[3]*self.num_modals, self.channels[3]),
        ])

    def forward(self, x, layer_idx):
        B, C, H, W = x[0].shape
        # conv fusion
        x = torch.cat(x, dim=1)
        x = x.flatten(2).transpose(1, 2)
        x_sum = self.liner_fusion_layers[layer_idx](x)
        return x_sum.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
class ChannelProcessing(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., drop_path=0., mlp_hidden_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_v = MLP(dim, mlp_hidden_dim)
        self.norm_v = norm_layer(dim)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W, atten=None):
        B, N, C = x.shape

        v = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = x.reshape(B, N, self.num_heads,  C // self.num_heads).permute(0, 2, 1, 3)

        q = q.softmax(-2).transpose(-1,-2)
        _, _, Nk, Ck  = k.shape
        k = k.softmax(-2)
        k = torch.nn.functional.avg_pool2d(k, (1, Ck))
        
        attn = self.sigmoid(q @ k)

        Bv, Hd, Nv, Cv = v.shape
        v = self.norm_v(self.mlp_v(v.transpose(1, 2).reshape(Bv, Nv, Hd*Cv), H, W)).reshape(Bv, Nv, Hd, Cv).transpose(1, 2)
        x = (attn * v.transpose(-1, -2)).permute(0, 3, 1, 2).reshape(B, N, C)
        return x 


mit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class Block_every_one(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))
        # Initialize adapters dynamically based on number of modalities
        for i in range(num_modalities):
            for j in range(num_modalities):  
                if i < j:
                    adap_t_att = Bi_direct_adapter(dim)  
                    adap_t_mlp = Bi_direct_adapter(dim)              
                    # ---------- saving in setattr
                    setattr(self, f"adap_t_att{i + 1}{j + 1}", adap_t_att)
                    setattr(self, f"adap_t_mlp{i + 1}{j + 1}", adap_t_mlp)

    def forward(self, inputs, H, W):
        
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeAtt = outputs
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 其次，使用adap_t适配器进行模态间影响
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeAtt[i]  # 提取出原始的没有经过multi-attention的
            # 使用adap_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    
                    if i < j:
                        adap_t_att = getattr(self, f"adap_t_att{i + 1}{j + 1}")
                    else:
                        adap_t_att = getattr(self, f"adap_t_att{j + 1}{i + 1}")

                    outputs[j] = xi + self.drop_path(adap_t_att(self.norm1(x_ori)))
        
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeMLP = outputs
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 再次，经过adap2_t适配器处理
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeMLP[i]
            # 使用adap2_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]

                    if i < j:
                        adap_t_mlp = getattr(self, f"adap_t_mlp{i + 1}{j + 1}")
                    else:
                        adap_t_mlp = getattr(self, f"adap_t_mlp{j + 1}{i + 1}")

                    outputs[j] = xi + self.drop_path(adap_t_mlp(self.norm2(x_ori)))

        return outputs

class Block_every_two(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))
 
        # Initialize adapters dynamically based on number of modalities
        for i in range(num_modalities):
            for j in range(num_modalities):  
                if i != j:
                    adap_t_att = Bi_direct_adapter(dim)  
                    adap_t_mlp = Bi_direct_adapter(dim)              
                    # ---------- saving in setattr
                    setattr(self, f"adap_t_att{i + 1}{j + 1}", adap_t_att)
                    setattr(self, f"adap_t_mlp{i + 1}{j + 1}", adap_t_mlp)

    def forward(self, inputs, H, W):
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeAtt = outputs
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 其次，使用adap_t适配器进行模态间影响
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeAtt[i]  # 提取出原始的没有经过multi-attention的
            # 使用adap_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    adap_t_att = getattr(self, f"adap_t_att{i + 1}{j + 1}")
                    outputs[j] = xi + self.drop_path(adap_t_att(self.norm1(x_ori)))
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeMLP = outputs
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 再次，经过adap2_t适配器处理
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeMLP[i]
            # 使用adap2_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    adap_t_mlp = getattr(self, f"adap_t_mlp{i + 1}{j + 1}")
                    outputs[j] = xi + self.drop_path(adap_t_mlp(self.norm2(x_ori)))
        return outputs


class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0. , num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

    def forward(self, inputs: list, H, W):
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return outputs

# for MCUBES AD AN 请您在复现论文的时候考虑 StitchFusion实际上在两个模态的时候Block_shared和Block_every_one是相同的。
# 一些 .pth文件 是只能适配 Block_shared的。请您到时候辨别性的复现。
class Block_shared(nn.Module): 
    def __init__(self, dim, head, sr_ratio=1, dpr=0. , num_modalities=2, fan=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4)) if not fan else ChannelProcessing(dim, mlp_hidden_dim=int(dim * 4))

        self.adap_t = Bi_direct_adapter(dim)
        self.adap2_t = Bi_direct_adapter(dim)

    def forward(self, inputs: list, H, W):
        outputs = [x.clone() for x in inputs]
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeAtt = outputs
        # 首先，每个输入独立经过attention处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # 其次，使用adap_t适配器进行模态间影响
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeAtt[i]  # 提取出原始的没有经过multi-attention的
            # 使用adap_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    outputs[j] = xi + self.drop_path(self.adap_t(self.norm1(x_ori)))
        # ---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------
        outputs_orig_beforeMLP = outputs
        # 每个输入独立经过MLP处理
        for i in range(len(inputs)):
            x = outputs[i]
            outputs[i] = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # 再次，经过adap2_t适配器处理
        for i in range(len(inputs)):
            x_ori = outputs_orig_beforeMLP[i]
            # 使用adap2_t适配器影响所有其他模态
            for j in range(len(inputs)):
                if i != j:
                    xi = outputs[j]
                    outputs[j] = xi + self.drop_path(self.adap2_t(self.norm2(x_ori)))
        return outputs

# ----------------------------------------- 总体框架 ----------------------------------------- #
class stitchfusion(nn.Module):
    def __init__(self, model_name: str = 'B0', modals: list = ['rgb', 'depth', 'event', 'lidar']):
        super().__init__()
        assert model_name in mit_settings.keys(), f"Model name should be in {list(mit_settings.keys())}"
        embed_dims, depths = mit_settings[model_name]
        self.modals = modals[1:] if len(modals)>1 else []  
        self.num_modals = len(self.modals)
        drop_path_rate = 0.1
        self.channels = embed_dims
        self.num_stages = 4

        # ----------------------------------------------------------------- 定义RGB  
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        pano_1 = [1,2,5,8]
        pano_2 = [8,4,2,1]

        for i in range(self.num_stages):
            if i == 0 :
                cur = 0
            else:
                cur += depths[i-1]
            patch_embed = PatchEmbed(3 if i==0 else embed_dims[i-1], embed_dims[i], 7 if i == 0 else 3, 4 if i == 0 else 2, 7//2 if i == 0 else 3//2)
                
            
            # block = nn.ModuleList([Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) for j in range(depths[i])])

            # 1
            block = nn.ModuleList([Block_every_one(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])
            # # 2
            # if i < 2 :
            #     block = nn.ModuleList([Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) for j in range(depths[i])])
            # else:
            #     block = nn.ModuleList([Block_every_one(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])
            
            # 3
            # block = nn.ModuleList([Block_every_two(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])

            # # 4
            # if i < 2 :
            #     block = nn.ModuleList([Block(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j]) for j in range(depths[i])])
            # else:
            #     block = nn.ModuleList([Block_every_two(embed_dims[i], pano_1[i], pano_2[i], dpr[cur+j], self.num_modals+1) for j in range(depths[i])])
            
            norm = nn.LayerNorm(embed_dims[i])
            
            # ---------- saving in setattr
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

        feature_cross = FeatureCross(self.channels, num_modals = self.num_modals + 1)
        setattr(self, f"feature_cross", feature_cross)
        
        # feature_conc = FeatureConc(self.channels, num_modals = self.num_modals + 1)
        # setattr(self, f"feature_conc", feature_conc)

    def forward(self, x: list) -> list:
        
        x_in = [t.clone() for t in x]
        B = x[0].shape[0]
        outs = []

        for i in range(self.num_stages):         
            # -------------- Transformer -------------- #
            # patch_embed
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            for kk in range(len(x)):
                x_in[kk], H, W = patch_embed(x_in[kk])
            # block
            block = getattr(self, f"block{i + 1}")
            for blk_num in range(len(block)): 
                x_in = block[blk_num](x_in, H, W)
            # norm
            norm = getattr(self, f"norm{i + 1}")
            for kk in range(len(x)):
                x_in[kk] = norm(x_in[kk]).reshape(B, H, W, -1).permute(0, 3, 1, 2)
                
            #--------------- 收集融合特征 ---------------#
            # 1. 使用FFMs
            feature_cross = getattr(self, f"feature_cross")
            x_fusion = feature_cross(x_in, layer_idx=i)
            
            # 2. add 
            # x_fusion = torch.sum(torch.stack(x_in), dim=0)

            # -------------- saving 
            outs.append(x_fusion)

        return outs


if __name__ == '__main__':
    modals = ['img', 'aolp', 'dolp', 'nir']
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    model = stitchfusion('B2', modals)
    outs = model(x)
    for y in outs:
        print(y.shape)

