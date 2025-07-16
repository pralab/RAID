# RAID
Source code for the paper "RAID: A Dataset for Testing the Adversarial
Robustness of AI-Generated Image Detectors"

# Setup

```
conda create -n myenv python=3.10
conda activate myenv
pip install -r requirements.txt
```

# Detector Training

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
            ├── gen_0
            |      .
            |      .
      │── val   	
            │── real
            ├── gen_0
            |      .
            |      .
``` 

Then run the ```train_detectors.sh``` script

# Evaluating and attacking a detector
First, create a directory `ckpt` in the project root folder and download the model checkpoints for the different detectors into it, which you can find [here](https://huggingface.co/aimagelab/RAID_ckpt)!
Then, proceed to download the RAID dataset from the following [link](https://huggingface.co/datasets/aimagelab/RAID) and check that the structure matches the one below:

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

## Running the attack
- `run_attack.sh`: Run the ensemble attack and evaluate the provided model on adversarial examples. Optionally saves adversarial examples for dataset creation.

```
  --generate                  setting this argument generates and saves the adversarial dataset at the provided output_dir        
  --eval_model                model to be evaluated
  --eval_output_dir           directory where the evaluation results are saved
  --models                    model(s) to be attacked in the attack
  --device                    device on which to run the models. If a list is passed, its length must be equal to the number of models                   
  --path_to_dataset           path to the dataset
  --dataset_type              subfolders for the datasets with the same structure as the ELSA D3 dataset, wang2020 for the structure used by detectors' dataset such as Forensynths
  --output_dir                directory where the adversarial dataset is saved
  --epsilon                   the perturbation budget for the attack
  --num_steps                 number of steps for the attack
  --step_size                 attack step size
  --ensembling_strategy       the ensembling strategy to be used in the attack
  --ensemble_loss             the ensemble attack loss
```
## Evaluating on the adversarial dataset
- `run_experiments.sh`: Evaluate provided model(s) on a saved adversarial examples dataset.

```
  --model                     model to be evaluated
  --path_to_checkpoint        checkpoint of the model to be loaded
  --dataset_type              dataset for the adversarial datasetset generated, subfolders for the datasets with the same structure as the ELSA D3 dataset, wang2020 for the structure used by detectors' dataset such as Forensynths
  --output_dir           directory where the evaluation results are saved
```

## Evaluating and attacking additional detectors
[Evaluating and attacking an new detector](src/raid/models/example_model/model_wrapping.Md)

# File Structure

- ```raid/attacks/```: Directory containing the code for the ensemble attack and the used trackers and losses
- ```raid/datasets.py```: Python script containing the dataloaders for the datasets
- ```external/```: Directory containing essential files for loading and training the detectors
- ```raid/models/```: Directory containing the wrapped detectors
- ```raid/plots/plot_adv_example.py```: Python script used for plotting adversarial examples
- ```raid/scripts/```: Directory containing scripts to download the elsa D3 dataset from huggingface
- ```raid/attack_generate.py```: Python script to run the adversarial attack on an ensemble of detectors, evaluate it on a detector and generate the adversarial dataset
- ```raid/evaluate_detector.py```: Python script to evaluate detector(s) on a given dataset (adversarial or otherwise)
- ```run_attack.sh```: Script used for running the attack and evaluation
- ```run_experiments.sh```: Script for evaluation on a dataset
- ```train_detectors.sh```: Training script for a list of detectors

# Detector Categorization

Detector | Detection Method | Architecture | Dataset | Preprocessing | Performance 
--- | --- | --- | --- | --- | ---
Ojha2023 (Universal) | CLIP<br>Feature space not for AI-generated image + Trainable binary classifier | Pretrained CLIP:ViT-L/14 network + Trainable linear layer | ForenSynths (ProGAN, LSUN), DMs | Normalize (CLIP) + CenterCrop (224x224) | x
Corvi23 | Two methods:<br>1. CNN<br>2. Ensemble of two CNNs trained on different datasets | Modified ResNet50 with: No Downsampling | Custom (ProGAN, Latent Diffusion) | Normalize (ImageNet)  | x
Cavia2024 | CNN<br>Patch Level Scoring + Global Average Pooling | Modified ResNet50 with: Custom Convolutions | ForenSynths (ProGAN, LSUN) | Normalize (ImageNet) + ReSize (256x256) | x
Chen2024<br>(convnext) | Diffusion Reconstruction Contrastive Training (DRCT) framework utilizing contrastive/training loss on top of reconustructed images included during training |  ConvNeXt Architecture | DRCT-2M | Normalize (ImageNet) + CenterCrop (224) | x
Chen2024<br>(clip) | Diffusion Reconstruction Contrastive Training (DRCT) framework utilizing contrastive/training loss on top of reconustructed images included during training |  CLIP:ViT-L/14 Architecture | DRCT-2M | Normalize (ImageNet) + CenterCrop (224) | x
Koutlis2024 | CLIP<br>CLIP's intermediate encoder-block representations  | Pretrained CLIP:ViT-B/16 model + Trainable linear layers | ForenSynths (ProGAN, LSUN), Ojha, Tan | Normalize (CLIP) + CenterCrop (224) | x
Wang2020 | CNN<br>Pretrained ResNet50 on ImageNet trained for binary classification | ResNet50 | ForenSynths (ProGAN, LSUN) | Normalize (ImageNet) | x

# Detectors and Reference Papers

Detector | Reference Paper | Repository
--- | --- | ---
Ojha2023 (Universal) | [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174) | [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
Corvi23 | [On the detection of synthetic images generated by diffusion models](https://arxiv.org/abs/2211.00680) | [DMimageDetection](https://github.com/grip-unina/DMimageDetection)
Cavia2024 | [Real-Time Deepfake Detection in the Real-World](https://arxiv.org/abs/2406.09398) | [RealTime-DeepfakeDetection-in-the-RealWorld](https://github.com/barcavia/RealTime-DeepfakeDetection-in-the-RealWorld)
Chen2024<br>(convnext/clip)| [DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images](https://proceedings.mlr.press/v235/chen24ay.html) | [DRCT](https://github.com/beibuwandeluori/DRCT)
Koutlis2024 | [Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection](https://arxiv.org/abs/2402.19091) | [rine](https://github.com/mever-team/rine)
Wang2020 | [CNN-generated images are surprisingly easy to spot...for now](https://arxiv.org/abs/1912.11035) | [CNNDetection](https://github.com/PeterWang512/CNNDetection)


# Licenses
The provided MIT License only applies to the `raid` directory. The code
contained in the `external` folder, provided by third-parties and modified in
some parts, has its own licenses that are included in each subfolder.