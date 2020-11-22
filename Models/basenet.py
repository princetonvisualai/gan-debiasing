"""ResNet

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

[2] https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class ResNet50(nn.Module):    
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs, features

class ResNet18(nn.Module):    
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(512, hidden_size)
        self.fc = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        outputs = self.fc(self.dropout(self.relu(features)))

        return outputs, features
    
class ResNet50_base(nn.Module):   
    """ResNet50 but without the final fc layer"""
    
    def __init__(self, pretrained, hidden_size=2048, dropout=0.5):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)                
        self.resnet.fc = nn.Linear(2048, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)        

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)
        features = self.dropout(self.relu(features))

        return features


class ResNet50_GradCam(ResNet50):
    def __init__(self, n_classes=1, pretrained=True, hidden_size=2048, dropout=0.5):
        super(ResNet50_GradCam, self).__init__(n_classes=n_classes, pretrained=pretrained, hidden_size=hidden_size, dropout=dropout)
        
        self.gradients = None
    
    def activations_hook(self, grad):
        self.gradients = grad
                                                                                                                                                    
    def forward(self, x):
        x = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x))))
        features = self.resnet.layer4(self.resnet.layer3(self.resnet.layer2(self.resnet.layer1(x))))
        #features = self.resnet(x)
        h = features.register_hook(self.activations_hook)
        outputs = self.fc(self.dropout(self.relu(self.resnet.fc(self.resnet.avgpool(features).view(x.shape[0], features.shape[1])))))
        return features, outputs
    
    def get_activations_gradient(self):
        return self.gradients
    
