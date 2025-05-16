from attacks.loss import AvgEnsembleLoss
from torch.nn import CrossEntropyLoss

from models import (
    cavia2024,
    chen2024,
    corvi2023,
    koutlis2024,
    ojha2023,
    wang2020,
    EnsembleModel,
    vit_lp14_dinov2,
    vit_lp14_reg_dinov2,
    vit_lp16_siglip_384,
    vit_tp16_224_augreg_in21k,
)
from models.ensemble.ensemble_function import (
    RawEnsembleFunction,
    AvgEnsembleFunction,
    RandomEnsembleFunction
)

# models with default checkpoints
MODELS = {
    "cavia2024": (cavia2024, "ckpt/cavia2024/ForenSynth_LaDeDa.pth"),
    "chen2024_convnext": (
        chen2024,
        "ckpt/chen2024/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth",
    ),
    "chen2024_clip": (
        chen2024,
        "ckpt/chen2024/DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth"
    ),
    "corvi2023": (corvi2023, "ckpt/corvi2023/Grag2021_latent/model_epoch_best.pth"),
    "koutlis2024": (koutlis2024, "ckpt/koutlis2024/model_ldm_trainable.pth"),
    "ojha2023": (ojha2023, "ckpt/ojha2023/fc_weights.pth"),
    "wang2020": (wang2020, "ckpt/wang2020/blur_jpg_prob0.5.pth"),
    "vit_lp14_dinov2": (vit_lp14_dinov2, "ckpt/vit_lp14_dinov2/model_best.pth.tar"),
    "vit_lp14_reg_dinov2": (vit_lp14_reg_dinov2, "ckpt/vit_lp14_reg_dinov2/model_best.pth.tar"),
    "vit_lp16_siglip_384": (vit_lp16_siglip_384, "ckpt/vit_large_patch16_siglip_384/model_best.pth.tar"),
    "vit_tp16_224_augreg_in21k": (vit_tp16_224_augreg_in21k, "ckpt/vit_tp16_224_augreg_in21k/model_best.pth.tar"),
    "vit_tp16_224_code_augreg_in21k": (vit_tp16_224_augreg_in21k, "ckpt/vit_tp16_224_code_augreg_in21k/model_best.pth.tar"),

    "ModelEnsemble" : EnsembleModel,
}

ENSEMBLING_STRATEGIES = {
    "raw": RawEnsembleFunction,
    "avg": AvgEnsembleFunction,
    "random": RandomEnsembleFunction
}

ENSEMBLE_LOSSES = {
    "avg_ce": AvgEnsembleLoss(),
    "ce" : CrossEntropyLoss(reduction="none"),
}
