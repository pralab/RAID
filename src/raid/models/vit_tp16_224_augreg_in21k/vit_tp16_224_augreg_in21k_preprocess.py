import torch
from PIL import Image
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision import transforms


class Vit_tp16_224_augreg_in21kPreProcess(DataProcessing):
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                #https://github.com/google-research/vision_transformer/issues/210
                transforms.Lambda(lambda x: x * 2 - 1) # input is tranformed instead of using precomputed mean and std
            ]
        )

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in x])
        
    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
