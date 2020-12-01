from __future__ import absolute_import

import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import cv2
import argparse
import utils
import parse_args
import pickle
from Models.attr_classifier import attribute_classifier
import torchvision.transforms as T
from PIL import Image
from os import path

def generate_orig_images(model, num_images):

    noise, _ = model.buildNoiseData(num_images)
    
    print('Generating new images. Latent vectors stored at record/GAN_model/latent_vectors.pkl')
    #Saving latent vectors
    with open('record/GAN_model/latent_vectors.pkl', 'wb+') as f:
        pickle.dump(noise.detach().cpu().numpy(), f)
    
    out_dir = 'data/fake_images/AllGenImages/'
    utils.make_dir('data/fake_images')
    utils.make_dir(out_dir)

    batch_size=64
    N = int(num_images/batch_size)
    if num_images%batch_size!=0:
        N+=1
    count=0
    for ell in range(N):
        with torch.no_grad():
            generated_images = model.test(noise[ell*batch_size:(ell+1)*batch_size])

        for i in range(generated_images.shape[0]):
            grid = torchvision.utils.save_image(generated_images[i].clamp(min=-1, max=1), 
                out_dir+'gen_'+str(count)+'.jpg', 
                padding=0, 
                scale_each=True, 
                normalize=True)
            count+=1
    
    print('All images generated')


def generate_pair_images(model, latent_vectors, out_dir):
    

    print('Generating image pairs.')
    noise = torch.Tensor(latent_vectors)
    
    utils.make_dir(out_dir)

    batch_size=64
    
    num_images = noise.shape[0]
    N = int(num_images/batch_size)
    if num_images%batch_size!=0:
        N+=1
    count = 0
    for ell in range(N):
        with torch.no_grad():
            generated_images = model.test(noise[ell*batch_size:(ell+1)*batch_size])
        for i in range(generated_images.shape[0]):
            grid = torchvision.utils.save_image(generated_images[i].clamp(min=-1, max=1), 
                out_dir+'gen_'+str(count)+'.jpg', 
                padding=0, 
                scale_each=True, 
                normalize=True)
            count+=1
    
    print('All images generated.')


if __name__=="__main__":

    use_gpu = True if torch.cuda.is_available() else False

    
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                           'PGAN', model_name='celeba',
                           pretrained=True, useGPU=use_gpu)

    opt = parse_args.collect_args_generate()
    
    # Uncomment if deterministic run required
    
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.manual_seed(opt['random_seed'])
    #np.random.seed(opt['random_seed'])
    
    if opt['experiment']=='orig':
        generate_orig_images(model, opt['num_images'])
    elif opt['experiment']=='pair':
        latent = pickle.load(open(opt['latent_file'], 'rb'))
        generate_pair_images(model, latent, opt['save_dir'])





