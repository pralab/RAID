from torch import nn
import torch
import numpy as np
class FcDetectorUniversal(nn.Module):
    def __init__(self):
        super(FcDetectorUniversal, self).__init__()
        self.fc = self.fc = nn.Linear(768, 1)

    def forward(self, x):
        return self.fc(x)

    def score(self, features, labels):
        predictions = self.predict(features)
        return np.mean(predictions == labels)

    def predict(self, features, batch_size=32):
        with torch.no_grad():
            y_true, y_pred = [], []
            in_tens = torch.from_numpy(features).cuda()
            in_tens = in_tens.split(batch_size)
            for feature in in_tens:
                feature = feature.cuda()
                y_pred.extend(self.forward(feature).sigmoid().flatten().tolist())
            # y_pred.extend(self.forward(in_tens).sigmoid().flatten().tolist())
        y_pred = np.array(y_pred) > 0.5
        return y_pred