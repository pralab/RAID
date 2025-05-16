'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
''' 

import os
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
from torchvision.datasets import ImageFolder
from PIL import ImageFile, Image
from .processing import make_processing, add_processing_arguments
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset


class PathNameDataset(ImageFolder):
    def __init__(self, **keys):
        super().__init__(**keys)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"img": sample, "target": target, "path": path}


class FlatStructureDataset(Dataset):
    def __init__(self, root, transform=None, target_class=0):
        self.root = root
        self.transform = transform
        self.target_class = target_class
        
        self.image_paths = []
        for filename in os.listdir(root):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(root, filename))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
            
        return {"img": image, "target": self.target_class, "path": path}


def get_dataset(opt, dataroot):
    dset_lst = []
    transform = make_processing(opt)
    
    if os.path.isdir(os.path.join(dataroot, "real")):
        real_dir = os.path.join(dataroot, "real")
        if os.path.isdir(real_dir):
            real_dset = FlatStructureDataset(root=real_dir, transform=transform, target_class=0)
            print("#real images %6d in %s" % (len(real_dset), real_dir))
            dset_lst.append(real_dset)
        
        for item in os.listdir(dataroot):
            if item == 'real':
                continue
            fake_dir = os.path.join(dataroot, item)
            if os.path.isdir(fake_dir):
                fake_dset = FlatStructureDataset(root=fake_dir, transform=transform, target_class=1)
                print("#fake images %6d in %s" % (len(fake_dset), fake_dir))
                dset_lst.append(fake_dset)
    else:
        classes = os.listdir(dataroot)
        for cls in classes:
            root = os.path.join(dataroot, cls)
            if os.path.isdir(root):
                dset = PathNameDataset(root=root, transform=transform)
                print("#images %6d in %s" % (len(dset), root))
                dset_lst.append(dset)

    return torch.utils.data.ConcatDataset(dset_lst)


def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend([d.target_class] * len(d))
        #targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1.0 / torch.tensor(ratio, dtype=torch.float)
    if torch.all(w==w[0]):
        print(f"RandomSampler: # {ratio}")
        sampler = RandomSampler(dataset, replacement = False)
    else:
        w = w / torch.sum(w)
        print(f"WeightedRandomSampler: # {ratio}, Weightes {w}")
        sample_weights = w[targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights)
        )
    return sampler


def add_dataloader_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for dataloader
    parser.add_argument(
        "--dataroot", type=str, help="Path to the dataset to use during training"
    )
    # The path containing the train and the validation data to train on

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_threads", default=11, type=int, help='# threads for loading data')
    parser = add_processing_arguments(parser)
    return parser


def create_dataloader(opt, subdir='.', is_train=True):
    dataroot = os.path.join(opt.dataroot, subdir)
    dataset = get_dataset(opt, dataroot)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        sampler=get_bal_sampler(dataset) if is_train else None,
        num_workers=int(opt.num_threads),
    )
    return data_loader
