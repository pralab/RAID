import os
from PIL import Image
import numpy as np
from io import BytesIO
from tqdm import tqdm

def png2jpg(img, quality=75):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality)  # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)

path = ...
folder = "real" # flux - sd3 - gen_0 - gen_1 - gen_2 - gen_3 - real
complete_folder = os.path.join(path, folder)

for el in tqdm(os.listdir(complete_folder)):
    image_path = os.path.join(complete_folder, el)
    image_jpg = png2jpg(Image.open(image_path))
    
    #  save the image
    image_jpg.save(image_path.replace(".png", ".jpg"), "JPEG")
    # remove the original png image
    os.remove(image_path)

print("done!")
