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
from load_data import *
import argparse
import utils
import parse_args
import pickle
from sklearn.metrics import average_precision_score

def main(opt):
    attr_list = utils.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    

    #print(attr_name)
    print(opt)

    if opt['experiment']=='baseline':
        train = create_dataset_actual(
            opt['data_setting']['path'],
            opt['data_setting']['attribute'],
            opt['data_setting']['protected_attribute'],
            opt['data_setting']['params_real_train'],
            opt['data_setting']['augment'],
            CelebaDataset,
            number=opt['number'])

        val = create_dataset_actual(
            opt['data_setting']['path'],
            opt['data_setting']['attribute'],
            opt['data_setting']['protected_attribute'],
            opt['data_setting']['params_real_val'],
            False,
            CelebaDataset,
            split='valid')
        val_weight = None

        test = create_dataset_actual(
            opt['data_setting']['path'],
            opt['data_setting']['attribute'],
            opt['data_setting']['protected_attribute'],
            opt['data_setting']['params_real_val'],
            False,
            CelebaDataset,
            split='test')

    elif opt['experiment']=='model':
        train = create_dataset_all(
            opt['data_setting']['real_params'],
            opt['data_setting']['fake_params'],
            opt['data_setting']['params_train'],
            opt['data_setting']['augment'],
            CelebaDataset,
            split='train')

    elif opt['experiment']=='model_inv':
        train = create_dataset_inv(
            opt['data_setting']['real_params'],
            opt['data_setting']['fake_params'],
            opt['data_setting']['params_train'],
            opt['data_setting']['augment'],
            CelebaDataset,
            split='train')

    elif opt['experiment']=='fake_only':
        train = create_dataset_reflections(
            opt['data_setting']['fake_params'],
            opt['data_setting']['params_train'],
            opt['data_setting']['augment'],
            CelebaDataset)

    if opt['experiment'] in ['model', 'model_inv', 'fake_only']:
        val = create_dataset_actual(
            opt['data_setting']['real_params']['path'],
            opt['data_setting']['real_params']['attribute'],
            opt['data_setting']['real_params']['protected_attribute'],
            opt['data_setting']['params_val'],
            False,
            CelebaDataset,
            split='valid')

        val_weight = utils.compute_class_weight(val, opt['device'], opt['dtype']).cpu().numpy()

        test = create_dataset_actual(
            opt['data_setting']['real_params']['path'],
            opt['data_setting']['real_params']['attribute'],
            opt['data_setting']['real_params']['protected_attribute'],
            opt['data_setting']['params_val'],
            False,
            CelebaDataset,
            split='test')

    # Train the attribute classifier
    save_path = opt['save_folder']+'/best.pth'
    save_path_curr = opt['save_folder'] + '/current.pth'
    if not opt['test_mode']:
        print('Starting to train model...')
        model_path = None
        if path.exists(save_path_curr):
            print('Model exists, resuming training')
            model_path = save_path_curr
        AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=model_path)
        for i in range(AC.epoch, opt['total_epochs']):
            AC.train(train)
            acc = AC.check_avg_precision(val, weights = val_weight)
            if (acc>AC.best_acc):
                AC.best_acc = acc
                AC.save_model(save_path)
            AC.save_model(save_path_curr)

    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath=save_path)
    val_targets, val_scores = AC.get_scores(val)
    test_targets, test_scores = AC.get_scores(test)

    with open(opt['save_folder']+'/val_scores.pkl', 'wb+') as handle:
        pickle.dump(val_scores,handle)
    with open(opt['save_folder']+'/val_targets.pkl', 'wb+') as handle:
        pickle.dump(val_targets,handle)
    with open(opt['save_folder']+'/test_scores.pkl', 'wb+') as handle:
        pickle.dump(test_scores,handle)
    with open(opt['save_folder']+'/test_targets.pkl', 'wb+') as handle:
        pickle.dump(test_targets,handle)

    cal_thresh = utils.calibrated_threshold(val_targets[:, 0], val_scores)
    f1_score,f1_thresh = utils.get_threshold(val_targets[:, 0], val_scores)
    val_pred=np.where(val_scores>cal_thresh, 1, 0)
    test_pred=np.where(test_scores>cal_thresh, 1, 0)

    ap, ap_std = utils.bootstrap_ap(val_targets[:, 0], val_scores)
    deo, deo_std = utils.bootstrap_deo(val_targets[:, 1], val_targets[:, 0], val_pred)
    ba, ba_std = utils.bootstrap_bias_amp(val_targets[:, 1], val_targets[:, 0], val_pred)
    kl, kl_std = utils.bootstrap_kl(val_targets[:, 1], val_targets[:, 0], val_scores)

    val_results = {
        'AP':ap, 'AP_std': ap_std,
        'DEO':deo, 'DEO_std':deo_std,
        'BA':ba, 'BA_std': ba_std,
        'KL':kl, 'KL_std':kl_std,
        'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
        'opt': opt
    }
    
    print('Validation results: ')
    print('AP : {:.1f} +- {:.1f}', 100*ap, 200*ap_std)
    print('DEO : {:.1f} +- {:.1f}', 100*deo, 200*deo_std)
    print('BA : {:.1f} +- {:.1f}', 100*ba, 200*ba_std)
    print('KL : {:.1f} +- {:.1f}', kl, 2*kl)

    with open(opt['save_folder']+'/val_results.pkl', 'wb+') as handle:
        pickle.dump(val_results,handle)


    ap, ap_std = utils.bootstrap_ap(test_targets[:, 0], test_scores)
    deo, deo_std = utils.bootstrap_deo(test_targets[:, 1], test_targets[:, 0], test_pred)
    ba, ba_std = utils.bootstrap_bias_amp(test_targets[:, 1], test_targets[:, 0], test_pred)
    kl, kl_std = utils.bootstrap_kl(test_targets[:, 1], test_targets[:, 0], test_scores)

    test_results = {
        'AP':ap, 'AP_std': ap_std,
        'DEO':deo, 'DEO_std':deo_std,
        'BA':ba, 'BA_std': ba_std,
        'KL':kl, 'KL_std':kl_std,
        'f1_thresh': f1_thresh,
        'cal_thresh': cal_thresh,
        'opt': opt
    }
    
    print('Test results: ')
    print('AP : {:.1f} +- {:.1f}', 100*ap, 200*ap_std)
    print('DEO : {:.1f} +- {:.1f}', 100*deo, 200*deo_std)
    print('BA : {:.1f} +- {:.1f}', 100*ba, 200*ba_std)
    print('KL : {:.1f} +- {:.1f}', kl, 2*kl)

    with open(opt['save_folder']+'/test_results.pkl', 'wb+') as handle:
        pickle.dump(test_results,handle)


if __name__=="__main__":
    opt = parse_args.collect_args_main()

    main(opt)
