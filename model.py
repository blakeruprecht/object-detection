import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights

def get_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    transform = weights.transforms()
    return model, transform
