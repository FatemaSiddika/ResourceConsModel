import torch.nn as nn
from torchvision.models import resnet34, mobilenet_v2
import timm
import torch

def get_model(model_name, IsPreTrained, num_classes):
    if model_name == 'resnet34':
        model = resnet34(pretrained=IsPreTrained, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
    elif model_name == 'mobilenetv2':
        model = mobilenet_v2(pretrained=IsPreTrained, num_classes=num_classes)

    elif model_name == 'vit_base':
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=IsPreTrained,
            num_classes=num_classes,
            img_size=32,
            patch_size=4,
            in_chans=3,
            embed_dim=768
        )
        # Adjust positional embedding for 32x32 images
        num_patches = (32 // 4) ** 2
        model.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, 768))
        nn.init.trunc_normal_(model.pos_embed, std=0.02)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model