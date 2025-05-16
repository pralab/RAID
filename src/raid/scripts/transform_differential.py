import random
import torchvision.transforms as VT
import torchvision.transforms.functional as VF
import PIL
from PIL import Image
import torch


class Brightness:
    def __call__(self, img):
        min_val = 0.05
        max_val = 0.95
        # random.seed(0)
        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_brightness(img, v)

class Color:
    def __call__(self, img):
        min_val = 0.05
        max_val = 0.95
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_saturation(img, v)

class Contrast:
    def __call__(self, img):
        min_val = 0.05
        max_val = 0.95
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_contrast(img, v)

class Identity:
    def __call__(self, img):
        return img

class Rotate:
    def __call__(self, img):
        min_val = -30
        max_val = 30
        v = min_val + (max_val - min_val) * random.random()
        return VF.rotate(img, v)

class Sharpness:
    def __call__(self, img):
        min_val = 0.05
        max_val = 0.95
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.adjust_sharpness(img, v)

class ShearX:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[v, 0])

class ShearY:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        return VF.affine(img, angle=0, translate=[0, 0], scale=1, shear=[0, v])

class TranslateX:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        # random.seed(0)

        v = min_val + (max_val - min_val) * random.random()
        width = img.size[0] if isinstance(img, Image.Image) else img.shape[2]
        v = int(v * width)
        return VF.affine(img, angle=0, translate=[v, 0], scale=1, shear=[0, 0])

class TranslateY:
    def __call__(self, img):
        min_val = -0.3
        max_val = 0.3
        v = min_val + (max_val - min_val) * random.random()
        height = img.size[1] if isinstance(img, Image.Image) else img.shape[1]
        v = int(v * height)
        return VF.affine(img, angle=0, translate=[0, v], scale=1, shear=[0, 0])

class Solarize:
    def __call__(self, img):
        min_val = 0
        max_val = 1
        v = min_val + (max_val - min_val) * random.random()
        return VF.solarize(img, int(v * 1))  # Scale to 0-255 range for solarization

class GaussianBlur:
    def __call__(self, img):
        kernel_size = random.choice([(3, 3), (5, 5), (7, 7)])  # Randomly choose kernel size
        sigma = random.uniform(0.1, 2.0)  # Random sigma value
        return VF.gaussian_blur(img, kernel_size, sigma=[sigma])

class RandomRotation:
    def __call__(self, img):
        # random.seed(0)

        degrees = random.randint(-30, 30)  # Random rotation angle
        return VF.rotate(img, degrees)


class Cutout:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # only support for the multiple tenosr batch
        # Check if the input image is a PIL image
        is_pil = isinstance(img, Image.Image)
        if is_pil:
            w, h = img.size

            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)

            x1 = x + self.size
            y1 = y + self.size

            xy = (x, y, x1, y1)
            # color = (125, 123, 114)
            color = (0, 0, 0)
            img = img.copy()
            PIL.ImageDraw.Draw(img).rectangle(xy, color)

            return img
        else:
            assert len(img.shape) == 4
            # If the image is PIL, convert it to a tensor

            h, w = img.shape[-2], img.shape[-1]

            # Create a mask with ones
            mask = torch.ones((1, 1, h, w), dtype=torch.float32)

            # Randomly select coordinates for the cutout region
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)

            # Apply the cutout by setting the selected square region to 0
            mask[:, :, y:y + self.size, x:x + self.size] = 0

            # Apply the mask to the image
            img = img * mask.to(img.device)

            return img

# class ColorJitter:
#     def __call__(self, img):
#         # random.seed(0)
#
#         brightness = random.uniform(0.8, 1.2)
#         contrast = random.uniform(0.8, 1.2)
#         saturation = random.uniform(0.8, 1.2)
#         hue = random.uniform(-0.1, 0.1)
#         return VF.adjust_brightness(VF.adjust_contrast(VF.adjust_saturation(VF.adjust_hue(img, hue), saturation), contrast), brightness)


class RandomAffine:
    def __call__(self, img):
        # random.seed(0)

        # Set affine parameters
        degrees = random.randint(-30, 30)
        translate = (random.uniform(0.1, 0.3), random.uniform(0.1, 0.3))
        scale = (random.uniform(0.9, 1.1), random.uniform(0.9, 1.1))
        shear = random.uniform(-10, 10)

        # Determine image dimensions based on type
        if isinstance(img, Image.Image):
            width, height = img.size
        else:  # if img is a tensor
            height, width = img.shape[1], img.shape[2]

        # Calculate translation in pixels
        translate_px = [int(t * width) for t in translate]

        # Apply affine transformation
        return VF.affine(img, angle=degrees, translate=translate_px, scale=random.uniform(*scale), shear=[shear])




transform_list = [
        Brightness(),
        Color(),
        Contrast(),
        Identity(),
        Sharpness(),
        TranslateX(),
        TranslateY(),
        GaussianBlur(),
        RandomRotation(),
        Cutout(size=10),
        # ColorJitter(),
        RandomAffine(),
    ]

class RandAugment_differential:
    def __init__(self, n):
        self.n = n
        self.augment_list = transform_list  # Define this list with your augment functions

    def __call__(self, img):
        # random.seed(0)
        ops = random.sample(self.augment_list, k=self.n)

        # assert self.n == 3
        # img_1 = ops[0](img)
        # img_2 = ops[1](img_1)
        # img_3 = ops[2](img_2)
        #
        # return img_3

        for i, op in enumerate(ops):
            img = op(img)

            # # Save each iteration's image
            # if isinstance(img, Image.Image):  # Check if it's a PIL Image
            #     img.save(f"pil_iter_{i}.png")
            # else:  # Assume it's a tensor and convert to PIL before saving
            #     VF.to_pil_image(img).save(f"tensor_iter_{i}.png")

        return img