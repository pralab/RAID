import torch

from .clip import clip
from PIL import Image
import torch.nn as nn
import timm
CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768
}
import faiss
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

#TODO: put max distance for svm oneclass and probability calculation
MAX_distance = 6.037170510076637

class VITContrastive(nn.Module):
    def __init__(self, name, sk_lear_classifier, nn_classifier=None, num_classes=0):
        super(VITContrastive, self).__init__()
        self.nn_classifier = nn_classifier
        self.model = timm.create_model(name, num_classes=num_classes) # vit_tiny_patch16_224
        self.classifier = sk_lear_classifier

    def forward(self, x, return_feature=False, proba=False):
        features = self.model(x)
        if return_feature:
            return features
        if self.nn_classifier is not None:
            features = features.cpu().detach().numpy()
            faiss.normalize_L2(features)
            _, predictions = self.nn_classifier.search(features, 1)
            return predictions.tolist()
        features = features.cpu().detach().numpy()
        predictions = self.classifier.predict(features)
        if proba:
            if isinstance(self.classifier, OneClassSVM):
                return torch.from_numpy(MAX_distance - self.classifier.decision_function(features))
            if isinstance(self.classifier, KNeighborsClassifier):
                return torch.from_numpy(self.classifier.kneighbors(features)[0])
            if isinstance(self.classifier, LogisticRegression):
                return torch.from_numpy(self.classifier.predict_proba(features)[:,1])
        return torch.from_numpy(predictions)

