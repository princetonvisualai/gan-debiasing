from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
import torchvision.transforms as T
import pickle
import itertools
import utils
from sklearn.model_selection import train_test_split

class CelebaDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor()):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img = Image.open(ID).convert('RGB')
        #print(transform)
        X = self.transform(img)
        y = self.labels[ID]
        
        return X,y
        

def create_dataset_all(real_params, fake_params, params, augment, dataset, split='train'):

    list_ids = []
    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    labeldata = np.tile(labeldata, 2)
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))
    print(fake_params)
    domdata = domdata[fake_params['range_orig_image'][0]:fake_params['range_orig_image'][1]]
    domdata = np.concatenate([(1-domdata), domdata])
    print(domdata.shape)
    labels = {}
    label_val = fake_params['range_orig_label'][0]
    for i in range(fake_params['range_orig_image'][0], fake_params['range_orig_image'][1]):
        list_ids.append(fake_params['path_orig'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_orig'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[label_val],domdata[label_val]]) 
        label_val+=1
    if (label_val!=fake_params['range_orig_label'][1]):
        print('Help!')

    for i in range(fake_params['range_new'][0], fake_params['range_new'][1]):
        list_ids.append(fake_params['path_new'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_new'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], domdata[i]])
    

    img_path = real_params['path'] + '/img_align_celeba/'
    split_path = real_params['path'] + '/list_eval_partition_celeba.txt'
    attr_path = real_params['path'] + '/list_attr_celeba.txt'
    
    attribute = real_params['attribute']
    protected_attribute = real_params['protected_attribute']

    label_file = open(attr_path, 'r')
    label_file = label_file.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    number = real_params['number'] 
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    for i in range(beg+2, beg+ number+2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path+temp[0])
        labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])
        #labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  2*((int(temp[21])+1)//2)+(int(temp[40])+1)//2])
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    if augment:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
       
        
    dset = dataset(list_ids, labels, transform)
    loader = DataLoader(dset, **params)

    return loader


def create_dataset_reflections(fake_params, params, augment, dataset):

    list_ids = []
    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    labeldata = np.tile(labeldata, 2)
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))
    print(fake_params)
    domdata = domdata[fake_params['range_orig_image'][0]:fake_params['range_orig_image'][1]]
    domdata = np.concatenate([(1-domdata), domdata])
    print(domdata.shape)
    labels = {}
    label_val = fake_params['range_orig_label'][0]
    for i in range(fake_params['range_orig_image'][0], fake_params['range_orig_image'][1]):
        list_ids.append(fake_params['path_orig'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_orig'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[label_val],domdata[label_val]]) 
        label_val+=1
    if (label_val!=fake_params['range_orig_label'][1]):
        print('Help!')

    for i in range(fake_params['range_new'][0], fake_params['range_new'][1]):
        list_ids.append(fake_params['path_new'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_new'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], domdata[i]])
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    if augment:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
       
        
    dset = dataset(list_ids, labels, transform)
    loader = DataLoader(dset, **params)

    return loader

def create_dataset_only_correct(path_orig, path_new, range_orig_image, range_orig_label, range_new, label_path_orig, label_path_new, params, augment, dataset):

    list_ids = []
    labeldata_orig = pickle.load(open(label_path_orig, 'rb'))
    labeldata_new = pickle.load(open(label_path_new, 'rb'))
    #labeldata = np.tile(labeldata, 2)
    labels = {}
    label_val = range_orig_label[0]
    for (i_old, i_new) in itertools.zip_longest(range(range_orig_image[0], range_orig_image[1]), range(range_new[0], range_new[1])):
        if labeldata_orig[label_val]!=labeldata_new[i_new]:
            label_val+=1
            continue
        list_ids.append(path_orig + 'gen_'+str(i_old)+'.jpg')
        labels[path_orig + 'gen_'+str(i_old)+'.jpg'] = labeldata_orig[label_val]
        label_val+=1
        list_ids.append(path_new + 'gen_'+str(i_new)+'.jpg')
        labels[path_new + 'gen_'+str(i_new)+'.jpg'] = labeldata_new[i_new]

    if (label_val!=range_orig_label[1]):
        print('Help!')

    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    if augment:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
       
        
    dset = dataset(list_ids, labels, transform)
    loader = DataLoader(dset, **params)

    return loader

def create_dataset_actual(path, attribute, protected_attribute, params, augment, dataset, number=0, split='train', transform=None):
    
    img_path = path + '/img_align_celeba/'
    split_path = path + '/list_eval_partition_celeba.txt'
    attr_path = path + '/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    attr = {}
    for i in range(beg+2, beg+ number+2):
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        attr[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        if augment:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
       
    #print(transform)    
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader


def create_dataset_nolabel(path, range1, params, dataset):

    list_ids = []
    labels = {}
    for i in range(range1[0], range1[1]):
        list_ids.append(path + 'gen_'+str(i)+'.jpg')
        labels[path+ 'gen_'+str(i)+'.jpg'] = -1
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    transform = T.Compose([
        T.Resize(64),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
           
    dset = dataset(list_ids, labels, transform)

    loader = DataLoader(dset, **params)

    return loader


def create_dataset_full_skew(path, attribute1, attribute2, params, augment, dataset, opp = False, number=0, split='train', transform=None):
    
    img_path = path + '/img_align_celeba/'
    split_path = path + '/list_eval_partition_celeba.txt'
    attr_path = path + '/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    attr = {}
    use_val_pos = []
    use_val_neg = []

    for i in range(beg+2, beg+ number+2):
        temp = label[i].strip().split()
        if not opp:
            if int(temp[attribute1+1])== int(temp[attribute2+1]):
                if int(temp[attribute1+1])==1:
                    use_val_pos.append(i)
                else:
                    use_val_neg.append(i)
        else:    
            if int(temp[attribute1+1]) != int(temp[attribute2+1]):
                if int(temp[attribute1+1])==1:
                    use_val_pos.append(i)
                else:
                    use_val_neg.append(i)
                
    if len(use_val_pos)>len(use_val_neg):
        use_val_pos=np.random.choice(use_val_pos, len(use_val_neg))
    else:
        use_val_neg=np.random.choice(use_val_neg, len(use_val_pos))
    #print(len(use_val_pos), len(use_val_neg), len(use_val_pos+use_val_neg))
    use_pos = np.concatenate([use_val_pos, use_val_neg])
    for i in use_pos:
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        attr[img_path+temp[0]]= torch.Tensor([int((int(temp[attribute1+1])+1)/2), int((int(temp[attribute2+1])+1)/2)])

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        if augment:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
       
    #print(transform)    
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader


def create_dataset_stargan(path, attribute, protected_attribute, params, augment, dataset, number=0, split='train', transform=None):
    
    img_path = 'data/stargan_contrastives_128_test/'
    split_path = path + '/list_eval_partition_celeba.txt'
    attr_path = path + '/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    attr = {}
    img_val = 4
    for i in range(beg+2, beg+ number+1):
        temp = label[i].strip().split()
        target = int((int(temp[attribute+1])+1)/2)
        prot1 = int((int(temp[20+1])+1)/2)
        prot2 = int((int(temp[39+1])+1)/2)
        for t in range(4):
            if t==0:
                prot=int(prot1*2+prot2)
            elif t==1:
                prot=int((1-prot1)*2+prot2)
            elif t==2:
                prot=int(prot1*2+(1-prot2))
            else:
                prot=int((1-prot1)*2+(1-prot2))

            list_ids.append(img_path+'gen_{}.jpg'.format(str(img_val+t)))
            attr[img_path+'gen_{}.jpg'.format(str(img_val+t))]=torch.Tensor([target,prot])
        img_val+=4

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        if augment:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
       
    #print(transform)    
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader

def create_dataset_stargan_val(path, attribute, protected_attribute, params, augment, dataset, number=0, split='valid', transform=None):
    
    img_path = path + '/img_align_celeba/'
    split_path = path + '/list_eval_partition_celeba.txt'
    attr_path = path + '/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637
    
    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    attr = {}
    for i in range(beg+2, beg+ number+2):
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        prot1 = int((int(temp[20+1])+1)/2)
        prot2 = int((int(temp[39+1])+1)/2)
        prot=int(prot1*2+prot2)
        attr[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  prot])
    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        if augment:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.RandomCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            transform = T.Compose([
                T.Resize(64),
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize
            ])
       
    #print(transform)    
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader

