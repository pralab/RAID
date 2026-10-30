from constants import MODELS, ENSEMBLING_STRATEGIES, ENSEMBLE_LOSSES


if __name__ == '__main__':
    model = MODELS['ModelEnsemble'](
            {m: MODELS[m][0](MODELS[m][1], device="cuda:0")
             for i, m in enumerate(["cavia2024", 
                "chen2024_convnext",
                "chen2024_clip",
                "corvi2023",
                "koutlis2024",
                "ojha2023",
                "wang2020",
                "vit_lp14_dinov2",
                "vit_lp14_reg_dinov2",
                "vit_tp16_224_augreg_in21k",
                "vit_tp16_224_code_augreg_in21k"])},
            ENSEMBLING_STRATEGIES['raw'](None))