import argparse
import pprint
from pathlib import Path
import torch
from tqdm import tqdm
from raid.data import get_dataloader
import os
import time
import pickle
from PIL import Image
import numpy as np


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pprint.pp(vars(args))

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if not args.dry_run:
        ds_path = list(Path(args.path_to_adv_dataset).parts)
        ds_path.insert(-1, 'IMAGE_DATASET')
        output_dir = Path(*(ds_path[:-1]))

        if not (output_dir.exists()):
            output_dir.mkdir(exist_ok=True, parents=True)


    #########################################################################
    #                     Loading the dataset and models                    #
    #########################################################################

    _, adv_paths = get_dataloader(path=args.path_to_dataset,
                                  dataset_type='subfolders',
                                  batch_size=args.batch_size,
                                  device=device,
                                  subset=-1)

    adv_data_loaders, _ = get_dataloader(path=args.path_to_adv_dataset,
                                         dataset_type='dataset',
                                         batch_size=args.batch_size,
                                         device=device,
                                         subset=-1)

    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Start time: {start_time}")

    for (k, adv_data_loader), (_, img_paths) in tqdm(zip(adv_data_loaders.items(), adv_paths.items()), desc="Generating images..."):
        ds_path = list(Path(output_dir).parts)
        ds_path_sub = Path(img_paths[0]).parts[-2]
        ds_path.insert(len(ds_path), ds_path_sub)
        adv_output_dir = Path(*(ds_path))

        os.makedirs(os.path.join(output_dir, k), exist_ok=True)
        imgs = [img.cpu() for imgs, _ in adv_data_loader for img in imgs]

        for img, path in zip(imgs, img_paths):
            pil_image = Image.fromarray((img.cpu().numpy().transpose(1, 2, 0) * 255).astype('uint8'))
            file_name = f"{path.split('/')[-1].rsplit('.', 1)[0]}_adv.png"
            img_save_path = os.path.join(output_dir, k, file_name)

            pil_image.save(img_save_path)
            print(img_save_path)

    end_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"End time: {end_time}")

def parse_args():
    parser = argparse.ArgumentParser(description="Image dataset generation from pickle file")
    parser.add_argument("--path_to_dataset", help="Directory containing the normal dataset")
    parser.add_argument("--path_to_adv_dataset", help="Directory containing the adversarial dataset")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for data loader")
    parser.add_argument("--dry_run", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
