import torch
import torch.nn as nn
import torchvision


class teacher(nn.Module):
    def __init__(self, model_name='dinov2_vitl14', num_classes=10, pretrained=True):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)
        self.out_dim = self.backbone.embed_dim
        self.linear = nn.Linear(self.out_dim, num_classes)

        # freeze weight
        for para in self.backbone.parameters():
            para.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        output = self.linear(x)

        return output
