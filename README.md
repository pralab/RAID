# RAID
Source code for the paper "RAID: A Dataset for Testing the Adversarial
Robustness of AI-Generated Image Detectors"

# Setup

```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

Copy `src/env_set.sh.example` to `src/env_set.sh`, fill in your machine's
paths, and source it before running anything (all scripts read from these
env vars):
```
cp src/env_set.sh.example src/env_set.sh
# edit src/env_set.sh
source src/env_set.sh
```

# Detector Training

If you want to use the available pretrained models, you can directly skip to the next [step](#evaluating-and-attacking-a-detector): the checkpoints, stored in [this HuggingFace repository](https://huggingface.co/aimagelab/RAID_ckpt), will be automatically downloaded.

## Dataset Setup

First, create a directory `data` in the project root folder and download the [ELSA D3 dataset](https://huggingface.co/datasets/elsaEU/ELSA_D3) (or a subset):
```
python src/raid/scripts/elsa_downloader.py --split train --amount 200_000
python src/raid/scripts/elsa_downloader.py --split val
```

## Dataset Structure

The structure of the ELSA D3 dataset is as follows:
```
data
└── ELSA_D3					
      ├── train
            ├── real
            ├── sd_14
            ├── sd_21
            ├── sd_xl
            ├── df_if
      │── val   	
            │── real
            ├── sd_14
            ├── sd_21
            ├── sd_xl
            ├── df_if
            |      .
``` 

## Training
The ```train_detectors.sh``` file contains all the training scripts to be launched.
You need to uncomment the lines corresponding to the detectors you want to train before launching it.
Additionally, to use the new generated checkpoints, you have to modify the `constants.py` file to have,
for each model inside the `MODELS` dictionary, the respective checkpoint path.

# Evaluating and attacking a detector
Download the RAID dataset from the following [link](https://huggingface.co/datasets/aimagelab/RAID) and check that the structure matches the one below:

```
data
└── RAID					
      ├── original
            ├── real
            ├── gen_0
            |      .
            |      .
      │── epsilon32   	
            │── real
            ├── gen_0
            |      .
            |      .
```

# Reproducing Results

All commands are run from the repository root, after `source src/env_set.sh`.

## 1. Attack generation

`src/run_experiments.sh` is the single entry point for all GPU attack generation,
one instance per device:

```
./src/run_experiments.sh <device> [epsilon] [step_size] [num_steps] [--mode all|ablation|experiments]

# One (epsilon, device) budget, full pipeline (ablation + white-box + LOO + RAID):
./src/run_experiments.sh cuda:0 8/255  0.01 --mode all
./src/run_experiments.sh cuda:1 16/255 0.03 --mode all
./src/run_experiments.sh cuda:2 32/255 0.03 --mode all

# Just the PGD ablation sweep (step sizes x iterations x seeds) for one epsilon:
./src/run_experiments.sh cuda:0 --mode ablation --epsilon 8/255

# Just white-box (1000) + leave-one-out (1000) + full RAID ensemble (full dataset):
./src/run_experiments.sh cuda:0 8/255 0.01 --mode experiments
```

Each run writes the adversarial images as tensors to `adv_dataset.pkl`. To export them as PNGs
(one folder per class, `<name>_adv.png`):
```
python3 src/raid/scripts/generate_image_dataset.py \
    --path_to_dataset $TEST_PATH \
    --path_to_adv_dataset <adv dir>/adv_dataset.pkl \
    [--output_dir <dir>] [--device cpu]
```
PNGs are saved to `<adv dir>/IMAGE_DATASET/` unless `--output_dir` is given.

Noise baselines (no GPU needed):
```
python3 src/generate_noise.py --type all       # control (ablation) + baseline (gaussian/uniform)
```

## 2. Evaluation and metrics

`src/compute_metrics.py` evaluates detectors and (for ablation) computes LPIPS/ASR
and rebuilds the results CSV:

```
python3 src/compute_metrics.py --mode clean       --device cuda:0   # detectors on the unattacked test set
python3 src/compute_metrics.py --mode experiments --device cuda:0   # detectors on white-box/LOO/RAID outputs
python3 src/compute_metrics.py --mode ablation    --device cuda:0   # AUROC + LPIPS + ASR for the ablation grid
python3 src/compute_metrics.py --mode all         --device cuda:0   # all three, in order
```

```
python3 src/build_results_table.py       # print paper tables (default: eps 16, 32)
python3 src/build_results_table.py 8 16 32
python3 src/plot_auroc_vs_lpips.py       # AUROC/ASR vs LPIPS figures for the ablation grid
```

# Attack and Evaluation Scripts

## Running the attack
- `raid/attack_generate.py`: Run the ensemble attack and evaluate the provided model on adversarial examples. Optionally saves adversarial examples for dataset creation. `run_experiments.sh` wraps this for the standard white-box/LOO/RAID/ablation runs.

```
  --generate                  saves the adversarial dataset at the provided output_dir        
  --eval_model                model to be evaluated
  --models                    model(s) to be attacked
  --device                    device(s) to run the models on                   
  --path_to_dataset           path to the dataset
  --dataset_type              subfolders | dataset | wang2020
  --output_dir                directory where the adversarial dataset is saved
  --epsilon                   perturbation budget (e.g. 16/255)
  --num_steps                 PGD iteration count
  --step_size                 PGD step size
  --ensembling_strategy       raw | avg | random
  --ensemble_loss             avg_ce
```

## Evaluating on the adversarial dataset
- `evaluate_detector.py`: Evaluate a model on a saved adversarial dataset.

```
  --model                     model to be evaluated
  --path_to_dataset           path to adv_dataset.pkl or subfolder dataset
  --dataset_type              dataset | subfolders
  --output_dir                directory where the evaluation results are saved
```

## Adding new detectors
[Evaluating and attacking a new detector](src/raid/models/example_model/model_wrapping.Md)

# File Structure

- `raid/attacks/` — ensemble attack, trackers, losses
- `raid/data/datasets.py` — dataloaders
- `raid/models/` — wrapped detectors
- `raid/attack_generate.py` — run adversarial attack and generate dataset
- `raid/evaluate_detector.py` — evaluate detector on a dataset
- `raid/scripts/generate_image_dataset.py` — export an `adv_dataset.pkl` to PNGs
- `raid/generate_noise_baseline.py` — Gaussian/uniform noise baselines (CLI helper, wrapped by `generate_noise.py`)
- `external/` — third-party detector code
- `env_set.sh.example` — template for machine-specific paths; copy to `env_set.sh` (gitignored) and source it
- `run_experiments.sh` — single entry point for GPU attack generation: PGD ablation sweep, white-box, leave-one-out, full RAID ensemble
- `generate_noise.py` — noise-control (ablation) and noise-baseline (gaussian/uniform) generation
- `compute_metrics.py` — evaluate detectors (clean / experiments / ablation modes) and compute AUROC + LPIPS + ASR
- `build_results_table.py` — aggregate results into paper tables
- `plot_auroc_vs_lpips.py` — AUROC/ASR vs LPIPS figures for the ablation grid
- `train_detectors.sh` — train the 7 detectors on ELSA D3

# Detector Categorization

Detector | Detection Method | Architecture | Dataset | Preprocessing
--- | --- | --- | --- | ---
Ojha2023 | CLIP feature space + linear classifier | CLIP ViT-L/14 + linear | ForenSynths, DMs | CLIP normalize + CenterCrop 224
Corvi2023 | Modified ResNet50, no downsampling | ResNet50 | ProGAN, Latent Diffusion | ImageNet normalize
Cavia2024 | Patch-level scoring + global avg pooling | ResNet50, 1x1 convs | ForenSynths | ImageNet normalize + Resize 256
Chen2024 (ConvNeXt) | DRCT contrastive training | ConvNeXt | DRCT-2M | ImageNet normalize + CenterCrop 224
Chen2024 (CLIP) | DRCT contrastive training | CLIP ViT-L/14 | DRCT-2M | ImageNet normalize + CenterCrop 224
Koutlis2024 | CLIP intermediate blocks + projection | CLIP ViT-B/16 + linear | ForenSynths, Ojha, Tan | CLIP normalize + CenterCrop 224
Wang2020 | ResNet50 binary classifier | ResNet50 | ForenSynths | ImageNet normalize

# Detectors and Reference Papers

Detector | Paper | Repository
--- | --- | ---
Ojha2023 | [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174) | [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
Corvi2023 | [On the detection of synthetic images generated by diffusion models](https://arxiv.org/abs/2211.00680) | [DMimageDetection](https://github.com/grip-unina/DMimageDetection)
Cavia2024 | [Real-Time Deepfake Detection in the Real-World](https://arxiv.org/abs/2406.09398) | [RealTime-DeepfakeDetection](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)
Chen2024 | [DRCT: Diffusion Reconstruction Contrastive Training](https://proceedings.mlr.press/v235/chen24ay.html) | [DRCT](https://github.com/beibuwandeluori/DRCT)
Koutlis2024 | [Leveraging Representations from Intermediate Encoder-blocks](https://arxiv.org/abs/2402.19091) | [rine](https://github.com/mever-team/rine)
Wang2020 | [CNN-generated images are surprisingly easy to spot...for now](https://arxiv.org/abs/1912.11035) | [CNNDetection](https://github.com/PeterWang512/CNNDetection)

# Licenses
The provided MIT License only applies to the `raid` directory. The code
contained in the `external` folder, provided by third-parties and modified in
some parts, has its own licenses that are included in each subfolder.

# Acknowledgements
This research has been partially supported by the Horizon Europe projects [ELSA](https://elsa-ai.eu) (GA no. 101070617), [Sec4AI4Sec](https://www.sec4ai4sec-project.eu) (GA no. 101120393), and [CoEvolution](https://coevolution-project.eu/) (GA no. 101168560); and by [SERICS](https://serics.eu/) (PE00000014) and [FAIR](https://fondazione-fair.it/en/) (PE00000013) under the MUR NRRP funded by the EU-NGEU.
We also acknowledge the CINECA award under the ISCRA initiative, for the availability of high-performance computing resources and support.

<p align="center">
    <img src="imgs/FundedbytheEU.png" alt="Funded by EU" style="height:80px; vertical-align: middle;"/> &nbsp;&nbsp;
    <img src="imgs/elsa.jpg" alt="ELSA" style="height:80px; vertical-align: middle;"/> &nbsp;&nbsp;
    <img src="imgs/sec4AI4sec.png" alt="Sec4AI4Sec" style="height:80px; vertical-align: middle;"/>
</p>
<br>
<p align="center">
    <img src="imgs/CoEvolution_Logo.svg" alt="CoEvolution" style="height:80px; vertical-align: middle;"/> &nbsp;&nbsp;
    <img src="imgs/SERICS.png" alt="SERICS" style="height:80px; vertical-align: middle;"/>
</p>
<p align="center">
    <img src="imgs/SAFER_Logo.png" alt="SAFER" style="height:160px;  vertical-align: middle;"/>
</p>
