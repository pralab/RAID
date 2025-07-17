import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision import transforms


class ModelPreProcess(DataProcessing):
    """
        Main model preprocssing wrapper template for testing the RAID
        dataset on an external model.
        
        - Evaluating using the ensemble attack:
            In the case where transformations are done using libraries other
            then torchvision's transforms, it is necessary to check that the
            applied transformations are differentiable, otherwise, the use of
            an equivalent torchvision transformation is required.
        - Evaluating using the RAID dataset:
            Transformations applied can be non-differentiable.
    """
    def __init__(self):
        self.transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.transform(img) for img in x])

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
