import torch
import torch.nn as nn
import torchvision
import numpy as np
from Models.attr_classifier import attribute_classifier
from load_data import create_dataset_nolabel, CelebaDataset
import parse_args
import pickle
import utils

# Computes baseline scores for generated images. Scores are stored in out_file. 

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
     
    print('Computing scores using {}/{} classifier'.format(opt['model_dir'], opt['attr_name']))
    _, scores = AC.get_scores(loader, False)

    threshold = pickle.load(open(opt['model_dir']+'/'+opt['attr_name']+'/test_results.pkl', 'rb'))['f1_thresh']
    scores = np.where(scores>threshold, 1.0, 0.0)

    with open(opt['out_file'], 'wb+') as handle:
        pickle.dump(scores, handle)
