import os
import argparse
import torch
import utils
import pickle
from os import listdir, path, mkdir
import numpy as np

def collect_args_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', 
                        choices=[
                                 'baseline', 
                                 'model', 
                                 'model_inv',
                                 'fake_only',
                                ])
    
    parser.add_argument('--experiment_name', type=str, default='_')
    parser.add_argument('--real_data_dir', type=str, default='data/celeba')
    parser.add_argument('--fake_data_dir_orig', type=str, default='data/fake_images/AllGenImages/')
    parser.add_argument('--fake_data_dir_new', type=str, default='_')
    parser.add_argument('--fake_scores_target', type=str, default='_')
    parser.add_argument('--fake_scores_protected', type=str, default='_')
    parser.add_argument('--no_cuda', dest='cuda', action='store_false')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--num_train', type=int, default=160000)
    parser.add_argument('--number', type=int, default=0)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    opt = create_experiment_setting(opt)
    return opt

def create_experiment_setting(opt):

    # Uncomment if deterministic run required. 
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.manual_seed(opt['random_seed'])
    #np.random.seed(opt['random_seed'])
    
    attr_list = utils.get_all_attr()
    attr_name = attr_list[opt['attribute']]
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32
    opt['print_freq'] = 100
    opt['total_epochs'] = 20
    orig_save = 'record/'
    if opt['protected_attribute']!=20:
        orig_save+='protected'+attr_list[opt['protected_attribute']]+'/'
    utils.make_dir('record')
    utils.make_dir(orig_save)
    if opt['experiment_name']=='_':
        opt['save_folder'] = os.path.join(orig_save +opt['experiment'], 
                                          attr_name)
        utils.make_dir(orig_save+opt['experiment'])
        utils.make_dir(opt['save_folder'])
    else:
        opt['save_folder'] = orig_save+opt['experiment_name']+'/'+attr_name

        utils.make_dir(orig_save+opt['experiment_name'])
        utils.make_dir(opt['save_folder'])
    
    
    optimizer_setting = {
        'optimizer': torch.optim.Adam,
        'lr': 1e-4,
        'weight_decay': 0,
    }
    opt['optimizer_setting'] = optimizer_setting
    opt['dropout'] = 0.5
    
    if opt['experiment']=='baseline':
        
        params_real_train = {'batch_size': 32,
                 'shuffle': True,
                 'num_workers': 0}
        
        params_real_val = {'batch_size': 64,
                 'shuffle': False,
                 'num_workers': 0}
        data_setting = {
            'path': opt['real_data_dir'],
            'params_real_train': params_real_train,
            'params_real_val': params_real_val,
            'protected_attribute': opt['protected_attribute'],
            'attribute': opt['attribute'],
            'augment': True
        }
        opt['data_setting'] = data_setting
    
    elif opt['experiment'] == 'model' or opt['experiment']=='fake_only':
        
        if opt['fake_data_dir_new']=='_':
            if opt['protected_attribute']!=20:
                input_path_new = 'data/fake_images/protected'+attr_list[opt['protected_attribute']]+'/'+attr_name+'/'
            else:
                input_path_new = 'data/fake_images/{}/'.format(attr_name)
        else:
            input_path_new = opt['fake_data_dir_new']


        input_path_orig = opt['fake_data_dir_orig']
        #scores = 'data/fake_images/' + attr_name+'_scores.pkl'
        if opt['fake_scores_target']=='_':
            scores = 'data/fake_images/{}_scores.pkl'.format(attr_name)
        else:
            scores = opt['fake_scores_target']
        if opt['fake_scores_protected']=='_':
            domain = 'data/fake_images/all_' + attr_list[opt['protected_attribute']]+'_scores.pkl'
        else:
            domain = opt['fake_scores_protected']
        params_train = {'batch_size': 32,
                 'shuffle': True,
                 'num_workers': 0}
        
        params_val = {'batch_size': 64,
                 'shuffle': False,
                 'num_workers': 0}
        real_params = {
            'path': opt['real_data_dir'], 
            'attribute': opt['attribute'], 
            'protected_attribute': opt['protected_attribute'], 
            'number': 0
        }
        fake_params = {
            'path_new': input_path_new,
            'path_orig': input_path_orig,
            'attr_path': scores,
            'dom_path': domain,
            'range_orig_image': (15000, 175000),
            'range_orig_label': ( 160000, 320000),
            'range_new': (0, 160000),
        }
        data_setting = {
            'real_params': real_params,
            'fake_params': fake_params,
            'augment': True,
            'params_train': params_train, 
            'params_val': params_val
        }
        opt['data_setting'] = data_setting
    return opt


def collect_args_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', 
                        choices=[
                                 'orig', 
                                 'pair',
                                ], default='orig')
    
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--save_dir', type=str, default='_')
    parser.add_argument('--latent_file', type=str, default='_')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=175000)
    parser.add_argument('--number', type=int, default=0)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--protected_val', type=int, default=0)
    parser.add_argument('--attr_val', type=int, default=0)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    
    attr_list = utils.get_all_attr()
    opt['attr_name'] = attr_list[opt['attribute']]
    opt['prot_attr_name'] = attr_list[opt['protected_attribute']]
    opt['device'] = torch.device('cuda' if opt['cuda'] else 'cpu')
    opt['dtype'] = torch.float32
    
    if opt['experiment']=='pair' and opt['save_dir']=='_':
        opt['save_dir']='data/fake_images/{}/'.format(opt['attr_name'])
    if opt['experiment']=='pair' and opt['latent_file']=='_':
        opt['latent_file']='record/GAN_model/latent_vectors_{}.pkl'.format(opt['attr_name'])
    return opt


def collect_args_scores():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--model_dir', type=str, default='record/baseline')
    parser.add_argument('--out_file', type=str, default='_')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--num_images', type=int, default=175000)
    parser.add_argument('--number', type=int, default=0)
    #parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    attr_list = utils.get_all_attr()
    opt['attr_name'] = attr_list[opt['attribute']]
    if torch.cuda.is_available(): 
        opt['device'] = torch.device('cuda')
    else:
        opt['device'] = torch.device('cpu')
    opt['dtype'] = torch.float32
    if opt['out_file']=='_':
        opt['out_file']='data/fake_images/all_{}_scores.pkl'.format(opt['attr_name'])

    return opt


def collect_args_linear():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute', type=int, default=31)
    parser.add_argument('--protected_attribute', type=int, default=20)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--attr_val', type=int, default=0)
    parser.add_argument('--protected_val', type=int, default=0)
    parser.add_argument('--number', type=int, default=0)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    
    attr_list = utils.get_all_attr()
    opt['attr_name'] = attr_list[opt['attribute']]
    opt['prot_attr_name'] = attr_list[opt['protected_attribute']]
    if torch.cuda.is_available(): 
        opt['device'] = torch.device('cuda')
    else:
        opt['device'] = torch.device('cpu')
    opt['dtype'] = torch.float32

    return opt


def collect_args_full_skew():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attribute1', type=int, default=31)
    parser.add_argument('--attribute2', type=int, default=20)
    parser.add_argument('--real_data_dir', type=str, default='data/celeba')
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--test_mode', type=bool, default=False)
    parser.add_argument('--opp', type=bool, default=False)
    parser.set_defaults(cuda=True)

    opt = vars(parser.parse_args())
    
    attr_list = utils.get_all_attr()
    opt['attr_name1'] = attr_list[opt['attribute1']]
    opt['attr_name2'] = attr_list[opt['attribute2']]
    if torch.cuda.is_available(): 
        opt['device'] = torch.device('cuda')
    else:
        opt['device'] = torch.device('cpu')
    opt['dtype'] = torch.float32
    opt['total_epochs']=20 
    params_real_train = {'batch_size': 32,
             'shuffle': True,
             'num_workers': 0}
    
    params_real_val = {'batch_size': 64,
             'shuffle': False,
             'num_workers': 0}

       
    data_setting = {
        'path': opt['real_data_dir'],
        'params_real_train': params_real_train,
        'params_real_val': params_real_val,
        'attribute1': opt['attribute1'],
        'attribute2': opt['attribute2'],
        'augment': True
    }
    opt['data_setting'] = data_setting
    if opt['opp']:
        opt['save_folder'] = 'record/full_skew/attr_{}_{}_opp/'.format(opt['attribute1'],opt['attribute2'])
    else:
        opt['save_folder'] = 'record/full_skew/attr_{}_{}/'.format(opt['attribute1'],opt['attribute2'])
    utils.make_dir('record/full_skew')
    utils.make_dir(opt['save_folder'])
    return opt

