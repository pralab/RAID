from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .vit_tiny import VITContrastive

VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
    'ViTContrastive:vit_tiny_patch16_224',
]





def get_model(name, sklearn_classifier=None, nn_classifier=None, num_classes=0):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:], nn_classifier=nn_classifier)
    elif name.startswith("CLIP:"):
        return CLIPModel(name[5:], nn_classifier=nn_classifier)
    elif name.startswith("ViTContrastive:"):
        return VITContrastive(name[15:], sklearn_classifier, nn_classifier=nn_classifier, num_classes=num_classes)
    else:
        assert False 
