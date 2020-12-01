import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.optim as optim
import numpy as np
from Models.attr_classifier import attribute_classifier
from load_data import * 
import argparse
import utils
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
        opt['data_setting']['params_real_val'],
        False,
        CelebaDataset,
        split='valid',
        opp=opt['opp']) 
        
        
        
    
    save_path = opt['save_folder']+'/best.pth' 
    save_path_curr = opt['save_folder'] + '/epoch'
    if not opt['test_mode']:
        model_path = None 
        AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=model_path)
        val_weight = None 
        for i in range(AC.epoch, opt['total_epochs']):
            AC.train(train)
            acc = AC.check_avg_precision(val, weights = val_weight)
            if acc >AC.best_acc:
                AC.best_acc = acc
                AC.save_model(save_path)
            AC.save_model(save_path_curr+str(i)+'.pth')
    
    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=save_path)
    
    
    for attr in [opt['attribute1'], opt['attribute2']]:
        val = create_dataset_actual(
            opt['data_setting']['path'],
            attr,
            20,
            opt['data_setting']['params_real_val'],
            False,
            CelebaDataset,
            split='valid')

        val_targets, val_scores = AC.get_scores(val)

        with open(opt['save_folder']+'/val_scores_{}.pkl'.format(attr), 'wb+') as handle:
            pickle.dump(val_scores,handle)
        
        if opt['opp'] and attr==opt['attribute2']:
            val_targets = 1-val_targets

        print('AP for attribute {}: {}', attr, 100*average_precision_score(val_targets, val_scores))


if __name__=="__main__":
    opt = parse_args.collect_args_full_skew()
    main(opt)
