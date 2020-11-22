import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torch.nn.functional as F
import numpy as np
from os import listdir, path, mkdir
from PIL import Image
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from Models.attr_classifier import attribute_classifier
from Models.weighted_attr_classifier import weighted_attribute_classifier
from Models.fake_attr_classifier import fake_attribute_classifier
from Models.dom_ind_attr_classifier import dom_ind_attribute_classifier
from load_data import * 
import argparse
import utils
import parse_args
import pickle
from sklearn.metrics import average_precision_score

def main(opt):
    attr_list = utils.get_all_attr()
    attr_name1 = attr_list[opt['attribute1']]
    attr_name2 = attr_list[opt['attribute2']]
    
    
    print(attr_name1, attr_name2)
    print(opt)
    if opt['attribute1']==opt['attribute2']:
        return

    train = create_dataset_full_skew(
        opt['data_setting']['path'], 
        opt['attribute1'],
        opt['attribute2'],
        opt['data_setting']['params_real_train'],
        opt['data_setting']['augment'],
        CelebaDataset,
        opp=opt['opp'])

    val = create_dataset_full_skew(
        opt['data_setting']['path'], 
        opt['attribute1'],
        opt['attribute2'],
        opt['data_setting']['params_real_train'],
        False,
        CelebaDataset,
        split='valid',
        opp=opt['opp']) 
        
        
        
    
    save_path = opt['save_folder']+'/best.pth' 
    save_path_curr = opt['save_folder'] + '/epoch'
    if not opt['test_mode']:
        model_path = None 
        AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=model_path)
        val_weight = None #utils.compute_class_weight(val, opt['device'], opt['dtype']).cpu().numpy()
        #acc = AC.check_avg_precision(val_real)
        for i in range(AC.epoch, opt['total_epochs']):
            AC.train(train)
            acc = AC.check_avg_precision(val, weights = val_weight)
            if acc >AC.best_acc:
                AC.best_acc = acc
                AC.save_model(save_path)
            AC.save_model(save_path_curr+str(i)+'.pth')
    
    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=save_path)
    
    
    val_targets, val_scores = AC.get_scores(val)

    with open(opt['save_folder']+'/val_scores.pkl', 'wb+') as handle:
        pickle.dump(val_scores,handle)
    with open(opt['save_folder']+'/val_targets.pkl', 'wb+') as handle:
        pickle.dump(val_targets,handle)
    
    #print('VALIDATION results')
    #print('Weighted average precision = ', dict_results['Weighted_AP'], '+-', 2*dict_results['Weighted_AP_std'])
    #print('Average precision = ', dict_results['AP'], '+-', 2*dict_results['AP_std'], flush=True)
        
    #print('Best threshold = ', thresh)
    #print('Best f1 score = ', f_score, flush=True)
        
    
    
    #with open(opt['save_folder']+'/val_results.pkl', 'wb+') as handle:
    #    pickle.dump(dict_results,handle)
    


if __name__=="__main__":
    opt = parse_args.collect_args_full_skew()
    main(opt)
