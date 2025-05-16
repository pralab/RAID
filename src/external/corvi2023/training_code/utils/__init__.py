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

from .dataset import create_dataloader
from .training import TrainingModel
class EarlyStopping:
    def __init__(self, init_score=None, patience=1, verbose=False, delta=0):
        self.best_score = init_score
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.count_down = self.patience
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            if self.verbose:
                print(f'Score set to {score:.6f}.')
            self.best_score = score
            self.count_down = self.patience
            return True
        elif score <= self.best_score + self.delta:
            self.count_down -= 1
            if self.verbose:
                print(f'EarlyStopping count_down: {self.count_down} on {self.patience}')
            if self.count_down <= 0:
                self.early_stop = True
            return False
        else:
            if self.verbose:
                print(f'Score increased from ({self.best_score:.6f} to {score:.6f}).')
            self.best_score = score
            self.count_down = self.patience
            return True

    def reset_counter(self):
        self.count_down = self.patience
        self.early_stop = False
