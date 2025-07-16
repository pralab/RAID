import torch
from secmlt.models.data_processing.data_processing import DataProcessing


class ModelPostProcess(DataProcessing):
    """
        Main model postprocssing wrapper template for testing the RAID
        dataset on an external model.
        
        The wrapped model should return two values for the negative and
        positive class respectively: [negative, positive].
        For single output models with threshold 0, this can be done
        simply by returning [-x, x].
    """
    def _process(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([-x,x], dim=-1)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return NotImplemented
