import torch
from secmlt.models.data_processing.data_processing import DataProcessing
from external.aide.aide_build import AIDEFiveView


class AIDEPreProcess(DataProcessing):

    def __init__(self):
        self._view = AIDEFiveView().eval()
        self._device = None

    def _process(self, x: torch.Tensor) -> torch.Tensor:
        if self._device != x.device:
            self._view = self._view.to(x.device)
            self._device = x.device
        return self._view(x)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
