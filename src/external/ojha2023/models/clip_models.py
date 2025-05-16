from .clip import clip
import torch.nn as nn
import faiss

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-B/16": 512,
}

class CLIPModel(nn.Module):
    def __init__(self, name, num_classes=1, nn_classifier=None):
        super(CLIPModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu") # self.preprecess will not be used during training, which is handled in Dataset class 
        self.fc = nn.Linear( CHANNELS[name], num_classes )
        self.nn_classifier = nn_classifier
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        if self.nn_classifier is not None:
            features = features.cpu().detach().numpy()
            faiss.normalize_L2(features)
            _, predictions = self.nn_classifier.search(features, 1)
            return predictions.tolist()
        return self.fc(features)

