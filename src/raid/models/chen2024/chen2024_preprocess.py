import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from torchvision import transforms


class Chen2024PreProcess(DataProcessing):
    def __init__(self):
        """self.transform = A.Compose(
            [
                A.PadIfNeeded(
                   min_height=224,
                    min_width=224,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.CenterCrop(height=224, width=224),
                A.PadIfNeeded(
                    min_height=224,
                    min_width=224,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )"""
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        """transformed_images = []
        for img in x:
            img_np = img.permute(1, 2, 0).cpu().detach().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            transformed = self.transform(image=img_np)["image"]
            transformed_images.append(transformed)

        return torch.stack(transformed_images)"""
        return torch.stack([self.transform(img) for img in x])

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
