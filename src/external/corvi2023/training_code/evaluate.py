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
import torch
import numpy as np
import tqdm
import pandas
import argparse
from networks import create_architecture, count_parameters
from utils.processing import make_post, make_normalize
from torchvision.transforms import Compose
from PIL import Image

def evaluate(name, model, transform, batch_size, device, csv_input, csv_output):
    table = pandas.read_csv(csv_input)[['filename', ]]
    rootdataset = os.path.dirname(os.path.join('.', csv_input))
    table.loc[:, name] = np.nan
    with torch.no_grad():
        model.eval()
        batch_img = list()
        batch_id = list()
        last_index = list(table.index)[-1]
        for index in tqdm.tqdm(table.index, total=len(table)):
            filename = os.path.join(rootdataset, table.loc[index, 'filename'])
            batch_img.append(transform(Image.open(filename).convert('RGB')))
            batch_id.append(index)

            if (len(batch_id) >= batch_size) or (index==last_index): 
                batch_img = torch.stack(batch_img, 0)
                out_tens = model(batch_img.to(device)).cpu().numpy()[:, -1]
                while len(out_tens.shape) > 1:
                    out_tens = np.mean(out_tens, -1)

                for ii, logit in zip(batch_id, out_tens):
                    table.loc[ii, name] = logit

                batch_img = list()
                batch_id = list()

        assert len(batch_id) == 0
        assert len(batch_img) == 0
        
        os.makedirs(os.path.dirname(os.path.join('.', csv_output)), exist_ok=True)
        table.to_csv(csv_output, index=False)  # save the results as csv file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument(
        "--checkpoints_dir",
        default="./checkpoints/",
        type=str,
        help="Path to the dataset to use during training",
    )
    parser.add_argument(
        "--arch", type=str, default="res50nodown",
        help="architecture name"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--epoch",
        type=str,
        default='best',
        help="Whether the network is going to be trained",
    )
    parser.add_argument("--norm_type", type=str, default="resnet")  # normalization type
    parser.add_argument(
        "--resizeSize",
        type=int,
        default=-1,
        help="scale images to this size post augumentation",    
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="run on CPU",
    )
    parser.add_argument('--csv_input', type=str)
    parser.add_argument('--csv_output', type=str)
    opt = parser.parse_args()
    opt.cropSize = -1
    
    
    device = torch.device('cpu') if opt.no_cuda else torch.device('cuda:0')
    model = create_architecture(opt.arch, pretrained=True, num_classes=1).to(device)
    num_parameters = count_parameters(model)
    print(f"Arch: {opt.arch} with #parameters {num_parameters}")
    
    load_path = os.path.join(opt.checkpoints_dir, opt.name, 'model_epoch_%s.pth' % opt.epoch)

    print('loading the model from %s' % load_path)
    model.load_state_dict(torch.load(load_path, map_location=device)['model'])
    model.to(device)
    
    t_post = make_post(opt)
    transform = make_normalize(opt)
    if t_post: transform = Compose([t_post, transform])
    
    evaluate(opt.name, model, transform, opt.batch_size, device, opt.csv_input, opt.csv_output)
