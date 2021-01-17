import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
from os import listdir, path, mkdir
from PIL import Image
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from Models.basenet import ResNet50


class attribute_classifier():
    
    def __init__(self, device, dtype, modelpath = None, learning_rate = 1e-4):
        #print(modelpath)
        self.model = ResNet50(n_classes=1, pretrained=True)
        self.model.require_all_grads()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.device=device
        self.dtype = dtype
        self.epoch = 0
        self.best_acc = 0.0
        self.print_freq=100
        if modelpath!=None:
            A = torch.load(modelpath, map_location=device)
            self.model.load_state_dict(A['model'])#torch.load(modelpath,map_location=device) )
            if (self.device==torch.device('cuda')):
                self.model.cuda()
            self.optimizer.load_state_dict(A['optim'])
            self.epoch = A['epoch']
            self.best_acc = A['best_acc']
    def forward(self, x):
        out, feature = self.model(x)
        return out, feature
    
    def save_model(self, path):
        torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict(), 'epoch':self.epoch, 'best_acc':self.best_acc}, path)
    
    def train(self, loader, weighted=False, weight_dict=None):
        """Train the model for one epoch"""
        
        self.model.train()
        train_loss = 0
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device=self.device, dtype=self.dtype), targets.to(device=self.device, dtype=self.dtype)
            domain = targets[:, 1]
            targets = targets[:, 0]
            
            self.optimizer.zero_grad()
            outputs, _ = self.forward(images)
            lossbce = torch.nn.BCEWithLogitsLoss()
            loss = lossbce(outputs.squeeze(), targets) 
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            if self.print_freq and (i % self.print_freq == 0):
                print('Training epoch {}: [{}|{}], loss:{}'.format(
                      self.epoch, i+1, len(loader), loss.item()), flush=True)
        
        self.epoch += 1

    def check_avg_precision(self, loader, weights=None, print_out=True):
        y_all, pred_all = self.get_scores(loader)
        acc = average_precision_score(y_all[:, 0], pred_all, sample_weight=weights)
        
        if print_out:
            print('Avg precision all = {}'.format(acc))
        return acc
    

    def get_scores(self, loader, labels_present = True):
        if (self.device==torch.device('cuda')):
            self.model.cuda()
        self.model.eval()  # set model to evaluation mode
        acc = 0.0
        y_all = []
        scores_all = []
        with torch.no_grad():
            for (x, y) in loader:
                x = x.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                y = y.to(device=self.device, dtype=torch.long)
                
                scores, _ = self.model(x)
                scores = torch.sigmoid(scores).squeeze()
                
                y_all.append(y.detach().cpu().numpy())
                scores_all.append(scores.detach().cpu().numpy())
            y_all = np.concatenate(y_all)
            pred_all = np.concatenate(scores_all)
        
        return y_all, pred_all


        
