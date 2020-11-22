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
from os import path, listdir
from Models.attr_classifier import attribute_classifier
from load_data import create_dataset_nolabel, CelebaDataset
import parse_args
import pickle
import utils


if __name__=="__main__":
    
    opt = parse_args.collect_args_scores()
    params = {'batch_size': 64,
             'shuffle': False,
             'num_workers': 1}
    
    loader = create_dataset_nolabel(
        'data/fake_images/AllGenImages/',
        (0, 175000),
        params,
        CelebaDataset)
    
    
    AC = attribute_classifier(opt['device'], opt['dtype'], modelpath='{}/{}/best.pth'.format(opt['model_dir'], opt['attr_name']))
    

    _, scores = AC.get_scores(loader, False)

    threshold = pickle.load(open(opt['model_dir']+'/'+opt['attr_name']+'/test_results.pkl', 'rb'))['f1_thresh']
    scores = np.where(scores>threshold, 1.0, 0.0)

    print(scores.sum(), flush=True)
    with open(opt['out_file'], 'wb+') as handle:
        pickle.dump(scores, handle)
