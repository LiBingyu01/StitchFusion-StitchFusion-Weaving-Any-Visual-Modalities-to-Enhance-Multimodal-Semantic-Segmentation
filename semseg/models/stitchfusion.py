import torch
from torch.nn import functional as F

from semseg.models.base import BaseModel
from semseg.models.heads import SegFormerHead

class stitchfusion(BaseModel):
    def __init__(self, backbone: str = 'stitchfusion-B0', num_classes: int = 20, modals: list = ['img', 'aolp', 'dolp', 'nir']) -> None:
        super().__init__(backbone, num_classes, modals)
        self.decode_head = SegFormerHead(self.backbone.channels, 256 if 'B0' in backbone or 'B1' in backbone else 512, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: list) -> list:
        y = self.backbone(x)
        y = self.decode_head(y)
        y = F.interpolate(y, size=x[0].shape[2:], mode='bilinear', align_corners=False)
        return y

    def init_pretrained(self, pretrained: str = None) -> None:
        checkpoint = torch.load(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        if 'model' in checkpoint.keys():
            checkpoint = checkpoint['model']
        msg = self.backbone.load_state_dict(checkpoint, strict=False)
        del checkpoint
        

if __name__ == '__main__':
    modals = ['img', 'aolp', 'dolp', 'nir']
    model = stitchfusion('stitchfusion-B2', 25, modals)
    model.init_pretrained('checkpoints/pretrained/segformer/mit_b2.pth')
    x = [torch.zeros(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024), torch.ones(1, 3, 1024, 1024)*2, torch.ones(1, 3, 1024, 1024) *3]
    y = model(x)
    print(y.shape)
