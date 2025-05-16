import argparse
import mimetypes
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

from PIL import Image


def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception as e:
        return False


def main(args):
    output_dir = Path(args.output_dir + "/" + args.split)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    count = 0
    pbar = tqdm(desc=f"Downloading {args.dataset}/{args.split}", total=args.amount)
    for row in iter(dataset):
        if count == 0:
            (output_dir / "real").mkdir(exist_ok=True)
            for i in range(4):
                (output_dir / row[f"model_gen{i}"].replace("/", "--")).mkdir(
                    exist_ok=True
                )
        try:            
            response = requests.get(row["url"], timeout=2)
            content_type = response.headers["content-type"]
            extension = mimetypes.guess_extension(content_type).lower()
            if extension not in [".png", ".jpg", ".jpeg"]:
                raise Exception
            real_path = output_dir / "real" / (row["id"] + extension)
            # Skips images already downloaded
            if not (real_path.exists() and is_valid_image(real_path)):
                with open(real_path, "wb") as f:
                    f.write(response.content)
        except Exception:
            if args.verbose: print(f"Error retrieving {row['url']}, skipping row.")
            continue

        for i in range(4):
            fake_path = output_dir / row[f"model_gen{i}"].replace("/", "--") / (row["id"] + ".png")

            if not (fake_path.exists() and is_valid_image(fake_path)):
                row[f"image_gen{i}"].save(fake_path)

        count += 1
        pbar.update()
        if count == args.amount:
            break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="elsaEU/ELSA_D3")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default="data/ELSA_D3")
    parser.add_argument("--amount", type=int, default=200_000, help='''this refers to the number of rows as opposed to images,
                such that each row contains 4 fake images and 1 real image''')
    parser.add_argument("--verbose", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
