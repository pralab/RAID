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
from ..networks import create_architecture, count_parameters


def add_training_arguments(parser):
    # parser is an argparse.ArgumentParser
    #
    # This adds the arguments necessary for the training

    parser.add_argument(
        "--arch", type=str, default="res50nodown",
        help="architecture name"
    )
    parser.add_argument(
        "--checkpoints_dir",
        default="./checkpoints/",
        type=str,
        help="Path to the dataset to use during training",
    )
    parser.add_argument("--pretrain", type=str, default=None, help="pretrained weights")
    parser.add_argument(
        "--optim", type=str, default="adam", help="optim to use [sgd, adam]"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="initial learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight decay"
    )

    parser.add_argument(
        "--beta1", type=float, default=0.9, help="momentum term of adam"
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="run on CPU",
    )

    parser.add_argument(
        "--continue_epoch",
        type=str,
        default=None,
        help="Whether the network is going to be trained",
    )

    return parser

class TrainingModel(torch.nn.Module):

    def __init__(self, opt, subdir='.'):
        super(TrainingModel, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, subdir)
        self.device = torch.device('cpu') if opt.no_cuda else torch.device('cuda')
        self.model = create_architecture(opt.arch, pretrained=True,  num_classes=1)
        num_parameters = count_parameters(self.model)
        print(f"Arch: {opt.arch} with #trainable {num_parameters}")

        self.loss_fn = torch.nn.BCEWithLogitsLoss().to(self.device)
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        if opt.optim == "adam":
            self.optimizer = torch.optim.Adam(
                parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay
            )
        elif opt.optim == "sgd":
            self.optimizer = torch.optim.SGD(
                parameters, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        if opt.pretrain:
            self.model.load_state_dict(
                torch.load(opt.pretrain, map_location="cpu")["model"]
            )
            print("opt.pretrain ", opt.pretrain)
        if opt.continue_epoch is not None:
            self.load_networks(opt.continue_epoch)

        # Support for multi-gpu processing
        print("NUMBER OF CORES!", torch.cuda.device_count())
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] /= 10.0
            if param_group["lr"] < min_lr:
                return False
        return True

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def train_on_batch(self, data):
        self.total_steps += 1
        self.model.train()
        input = data['img'].to(self.device)
        label = data['target'].to(self.device).float()
        output = self.model(input)
        if len(output.shape) == 4:
            ss = output.shape
            loss = self.loss_fn(
                output,
                label[:, None, None, None].repeat(
                    (1, int(ss[1]), int(ss[2]), int(ss[3]))
                ),
            )
        else:
            loss = self.loss_fn(output.squeeze(1), label)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.cpu()

    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
        }

        torch.save(state_dict, save_path)

    # load models from the disk
    def load_networks(self, epoch):
        load_filename = 'model_epoch_%s.pth' % epoch
        load_path = os.path.join(self.save_dir, load_filename)

        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=self.device)

        self.model.load_state_dict(state_dict['model'])
        self.model.to(self.device)

        try:
            self.total_steps = state_dict['total_steps']
        except:
            self.total_steps = 0

        try:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        except:
            pass


    def predict(self, data_loader):
        model = self.model.eval()
        with torch.no_grad():
            y_true, y_pred, y_path = [], [], []
            for data in tqdm.tqdm(data_loader):
                img = data['img']
                label = data['target'].cpu().numpy()
                out_tens = model(img.to(self.device)).cpu().numpy()[:, -1]

                #print(out_tens, label)

                assert label.shape == out_tens.shape

                y_pred.extend(out_tens.tolist())
                y_true.extend(label.tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return y_true, y_pred
