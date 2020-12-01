from __future__ import print_function, division
import os
import glob
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import pickle
import itertools

class CelebaDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor()):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img = Image.open(ID)
        #print(transform)
        X = self.transform(img)
        y = self.labels[ID]

        return X,y

class CelebaDataset_no_normalize(Dataset):
    def __init__(self, list_IDs, labels):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        ID = self.list_IDs[index]
        img = Image.open(ID)


        transform = T.Compose([
            T.ToTensor(),
        ])

        X = transform(img)
        y = self.labels[ID]

        return X,y

def create_dataset_all(real_params, fake_params, params, augment, dataset, split='train'):

    list_ids = []
    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    labeldata = np.tile(labeldata, 2)
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))
    domdata = domdata[fake_params['range_orig_image'][0]:fake_params['range_orig_image'][1]]

    domdata = np.concatenate([(1-domdata), domdata])
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


    img_path = real_params['path']+'/img_align_celeba/'
    split_path = real_params['path']+'/list_eval_partition_celeba.txt'
    attr_path = real_params['path']+'/list_attr_celeba.txt'

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

    # +1 for column names, +1 for CelebA naming convention
    for i in range(beg+2, beg+number+2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path+temp[0])
        labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

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

def create_dataset_xprimeonly(real_params, fake_params, params, augment, dataset, split='train'):


    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    labeldata = labeldata[15000:]
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))
    domdata = domdata[15000:]
    domdata = 1 - domdata

    list_ids = []
    labels = {}
    for i in range(160000):
        list_ids.append(fake_params['path_new'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_new'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], domdata[i]])

    img_path = real_params['path']+'/img_align_celeba/'
    split_path = real_params['path']+'/list_eval_partition_celeba.txt'
    attr_path = real_params['path']+'/list_attr_celeba.txt'

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

    # +1 for column names, +1 for CelebA naming convention
    for i in range(beg+2, beg+number+2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path+temp[0])
        labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

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

def create_dataset_stargan(real_params, fake_params, params, augment, dataset, split='train'):

    with open('list_attr_celeba.txt', "r") as f:
        attr = pd.read_csv(f, delim_whitespace=True, header=1)
    label_data = (attr['Smiling'].values[:162770] == 1).astype(np.float32)

    list_ids = []
    labels = {}
    for i in range(162770):
        # CelebA
        img_path = 'data/celeba/img_align_celeba/{:06d}.jpg'.format(i+1)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

        # Gender flipped
        img_path = 'data/fake_images_raninv/Smiling/prime/gen_{}.jpg'.format(i)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

        # Young flipped
        img_path = 'data/fake_images_raninv/Smiling/young/gen_{}.jpg'.format(i)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

        # Both flipped
        img_path = 'data/fake_images_stargan/gen_{}.jpg'.format(i+162770*2)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

    print('list_ids', len(list(set(list_ids))))

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

def create_dataset_stargan_allsgd(real_params, fake_params, params, augment, dataset, split='train'):

    with open('list_attr_celeba.txt', "r") as f:
        attr = pd.read_csv(f, delim_whitespace=True, header=1)
    label_data = (attr['Smiling'].values[:162770] == 1).astype(np.float32)

    list_ids = []
    labels = {}
    for i in range(162770):
        # CelebA
        img_path = 'data/celeba/img_align_celeba/{:06d}.jpg'.format(i+1)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

        # Gender flipped
        img_path = 'data/fake_images_stargan/gen_{}.jpg'.format(i)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

        # Young flipped
        img_path = 'data/fake_images_stargan/gen_{}.jpg'.format(i+162770)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

        # Both flipped
        img_path = 'data/fake_images_stargan/gen_{}.jpg'.format(i+162770*2)
        list_ids.append(img_path)
        labels[img_path] = torch.Tensor([label_data[i], 0.])

    print('list_ids', len(list(set(list_ids))))

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

def create_dataset_trainsyn(real_params, fake_params, params, augment, dataset, split='train'):

    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))

    list_ids = []
    labels = {}

    for i in range(15000, 175000):
        list_ids.append(fake_params['path_orig'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_orig'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], domdata[i]])

    img_path = real_params['path']+'/img_align_celeba/'
    split_path = real_params['path']+'/list_eval_partition_celeba.txt'
    attr_path = real_params['path']+'/list_attr_celeba.txt'

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

    # +1 for column names, +1 for CelebA naming convention
    for i in range(beg+2, beg+number+2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path+temp[0])
        labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

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

def create_dataset_inv(real_params, fake_params, params, augment, dataset, split='train'):

    list_ids = []
    labels = {}

    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))
    domdata = 1 - domdata # Reverse the gender label for z'

    for i in range(162770):
        list_ids.append(fake_params['path_new'] + 'gen_'+str(i)+'.jpg')
        labels[fake_params['path_new'] + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], domdata[i]])

    img_path = real_params['path']+'/img_align_celeba/'
    split_path = real_params['path']+'/list_eval_partition_celeba.txt'
    attr_path = real_params['path']+'/list_attr_celeba.txt'

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

    # +1 for column names, +1 for CelebA naming convention
    for i in range(beg+2, beg+number+2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path+temp[0])
        labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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

def create_dataset_inv_dist(real_params, fake_params, params, augment, dataset, split='train'):

    list_ids = []
    labels = {}

    labeldata = pickle.load(open(fake_params['attr_path'], 'rb')) # GT labels for z
    domdata = pickle.load(open(fake_params['dom_path'], 'rb')) # GT labels for z

    for dirname in ['z0_minus1', 'z0_plus1']:
    #for dirname in ['z0_minus3', 'z0_minus2', 'z0_minus1', 'on_hyperplane', 'z0_plus1', 'z0_plus2', 'z0_plus3']:
        dir = fake_params['path_new'] + dirname + '/'
        for i in range(162770):

            if domdata[i] == 1 and dirname == 'z0_minus1':
                list_ids.append(dir + 'gen_'+str(i)+'.jpg')
                labels[dir + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], 0])
            if domdata[i] == 0 and dirname == 'z0_plus1':
                list_ids.append(dir + 'gen_'+str(i)+'.jpg')
                labels[dir + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], 1])

            #list_ids.append(dir + 'gen_'+str(i)+'.jpg')
            #if dirname in ['z0_minus3', 'z0_minus2', 'z0_minus1']:
            #    labels[dir + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], 0])
            #else:
            #    labels[dir + 'gen_'+str(i)+'.jpg'] = torch.Tensor([labeldata[i], 1])

    #print('list_ids', len(list_ids))

    img_path = real_params['path']+'/img_align_celeba/'
    split_path = real_params['path']+'/list_eval_partition_celeba.txt'
    attr_path = real_params['path']+'/list_attr_celeba.txt'

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

    # +1 for column names, +1 for CelebA naming convention
    for i in range(beg+2, beg+number+2):
        temp = label_file[i].strip().split()
        list_ids.append(img_path+temp[0])
        labels[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

    #print('list_ids', len(list_ids))

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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


def create_dataset_fakezonly(real_params, fake_params, params, augment, dataset, split='train'):

    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))

    list_ids = []
    labels = {}
    for i in range(0, 160000):
        list_ids.append(fake_params['path_orig'] + 'gen_' + str(i) + '.jpg')
        labels[fake_params['path_orig'] + 'gen_' + str(i) + '.jpg'] = torch.Tensor([labeldata[i], domdata[i]])

    print('\n', len(labels), 'images in fake z only\n')

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


def create_dataset_reflections(path_orig, path_new, range_orig_image, range_orig_label, range_new, label_path, params, augment, dataset):

    list_ids = []
    labeldata = pickle.load(open(label_path, 'rb'))
    labeldata = np.tile(labeldata, 2)
    labels = {}
    label_val = range_orig_label[0]
    for i in range(range_orig_image[0], range_orig_image[1]):
        list_ids.append(path_orig + 'gen_'+str(i)+'.jpg')
        labels[path_orig + 'gen_'+str(i)+'.jpg'] = labeldata[label_val]
        label_val+=1
    if (label_val!=range_orig_label[1]):
        print('Help!')

    for i in range(range_new[0], range_new[1]):
        list_ids.append(path_new + 'gen_'+str(i)+'.jpg')
        labels[path_new + 'gen_'+str(i)+'.jpg'] = labeldata[i]

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

def create_dataset_actual(path, attribute, protected_attribute, params, augment, dataset, number=0, split='train'):

    img_path = path+'/img_align_celeba/'
    split_path = path+'/list_eval_partition_celeba.txt'
    attr_path = path+'/list_attr_celeba.txt'
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

def create_dataset_actual_skew(path, attribute, protected_attribute, params, augment, dataset, skew=0.5, number=0, split='train'):

    attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs',  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    attr_name = attr_list[attribute]
    #attr_name = 'Gray_Hair' # Uncomment for Male
    print('skew', skew)

    glob_pattern = os.path.join('celeba', '*.jpg')
    image_filenames_all = sorted(glob.glob(glob_pattern))[:162770]

    with open("list_attr_celeba.txt", "r") as f:
        attr = pd.read_csv(f, delim_whitespace=True, header=1)
    g = (attr['Male'][:162770].values == 1).astype(np.float16)
    a = (attr[attr_name][:162770].values == 1).astype(np.float16)

    MT = np.array(image_filenames_all)[np.logical_and(g==1, a==1)]
    MF = np.array(image_filenames_all)[np.logical_and(g==1, a==0)]
    FT = np.array(image_filenames_all)[np.logical_and(g==0, a==1)]
    FF = np.array(image_filenames_all)[np.logical_and(g==0, a==0)]
    print('Before selection: MT {}, MF {}, FT {}, FF {}'.format(MT.shape[0], MF.shape[0], FT.shape[0], FF.shape[0]))

    np.random.seed(1)
    if attr_name == 'Gray_Hair' and skew == 0.9:
        FT = FT[np.random.choice(np.arange(946), size=661, replace=False)]
        FF = FF[np.random.choice(np.arange(93563), size=88844, replace=False)]
    elif attr_name == 'Gray_Hair' and skew == 0.7:
        MT = MT[np.random.choice(np.arange(5950), size=2207, replace=False)]
        FF = FF[np.random.choice(np.arange(93563), size=92302, replace=False)]
    elif attr_name == 'Gray_Hair' and skew == 0.5:
        MT = MT[np.random.choice(np.arange(5950), size=946, replace=False)]
    elif attr_name == 'High_Cheekbones' and skew == 0.9:
        MT = MT[np.random.choice(np.arange(20972), size=5853, replace=False)]
        FF = FF[np.random.choice(np.arange(41836), size=25256, replace=False)]
    elif attr_name == 'High_Cheekbones' and skew == 0.7:
        MF = MF[np.random.choice(np.arange(47289), size=44562, replace=False)]
        FT = FT[np.random.choice(np.arange(52673), size=48935, replace=False)]
        FF = FF[np.random.choice(np.arange(41836), size=16600, replace=False)]
    elif attr_name == 'High_Cheekbones' and skew == 0.5:
        FT = FT[np.random.choice(np.arange(52673), size=20972, replace=False)]

    print('After selection:  MT {}, MF {}, FT {}, FF {}'.format(MT.shape[0], MF.shape[0], FT.shape[0], FF.shape[0]))
    print()

    list_ids = []
    attr = {}
    for i in range(FF.shape[0]):
        list_ids.append(FF[i])
        attr[FF[i]] = torch.Tensor([0, 0])
    for i in range(FT.shape[0]):
        list_ids.append(FT[i])
        attr[FT[i]] = torch.Tensor([1, 0]) # Gray_Hair, High_Cheekbones
        #attr[FT[i]] = torch.Tensor([0, 0]) # Male
    for i in range(MF.shape[0]):
        list_ids.append(MF[i])
        attr[MF[i]] = torch.Tensor([0, 1]) # Gray_Hair, High_Cheekbones
        #attr[MF[i]] = torch.Tensor([1, 1]) # Male
    for i in range(MT.shape[0]):
        list_ids.append(MT[i])
        attr[MT[i]] = torch.Tensor([1, 1])

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize(64),
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize
    ])

    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader

def create_dataset_all_skewdata(real_params, fake_params, params, augment, dataset, skew=0.5, split='train'):

    list_ids = []
    labeldata = pickle.load(open(fake_params['attr_path'], 'rb'))
    labeldata = np.tile(labeldata, 2)
    domdata = pickle.load(open(fake_params['dom_path'], 'rb'))
    domdata = domdata[fake_params['range_orig_image'][0]:fake_params['range_orig_image'][1]]

    domdata = np.concatenate([(1-domdata), domdata])
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

    attr_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs',  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
    'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose',
    'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    attr_name = attr_list[real_params['attribute']]
    print(attr_name, 'skew', skew)

    glob_pattern = os.path.join('celeba', '*.jpg')
    image_filenames_all = sorted(glob.glob(glob_pattern))[:162770]

    with open("list_attr_celeba.txt", "r") as f:
        attr = pd.read_csv(f, delim_whitespace=True, header=1)
    g = (attr['Male'][:162770].values == 1).astype(np.float16)
    a = (attr[attr_name][:162770].values == 1).astype(np.float16)

    MT = np.array(image_filenames_all)[np.logical_and(g==1, a==1)]
    MF = np.array(image_filenames_all)[np.logical_and(g==1, a==0)]
    FT = np.array(image_filenames_all)[np.logical_and(g==0, a==1)]
    FF = np.array(image_filenames_all)[np.logical_and(g==0, a==0)]
    print('Before selection: MT {}, MF {}, FT {}, FF {}'.format(MT.shape[0], MF.shape[0], FT.shape[0], FF.shape[0]))

    np.random.seed(1)
    if attr_name == 'Gray_Hair' and skew == 0.9:
        FT = FT[np.random.choice(np.arange(946), size=661, replace=False)]
        FF = FF[np.random.choice(np.arange(93563), size=88844, replace=False)]
    elif attr_name == 'Gray_Hair' and skew == 0.7:
        MT = MT[np.random.choice(np.arange(5950), size=2207, replace=False)]
        FF = FF[np.random.choice(np.arange(93563), size=92302, replace=False)]
    elif attr_name == 'Gray_Hair' and skew == 0.5:
        MT = MT[np.random.choice(np.arange(5950), size=946, replace=False)]
    elif attr_name == 'High_Cheekbones' and skew == 0.9:
        MT = MT[np.random.choice(np.arange(20972), size=5853, replace=False)]
        FF = FF[np.random.choice(np.arange(41836), size=25256, replace=False)]
    elif attr_name == 'High_Cheekbones' and skew == 0.7:
        MF = MF[np.random.choice(np.arange(47289), size=44562, replace=False)]
        FT = FT[np.random.choice(np.arange(52673), size=48935, replace=False)]
        FF = FF[np.random.choice(np.arange(41836), size=16600, replace=False)]
    elif attr_name == 'High_Cheekbones' and skew == 0.5:
        FT = FT[np.random.choice(np.arange(52673), size=20972, replace=False)]

    print('After selection:  MT {}, MF {}, FT {}, FF {}'.format(MT.shape[0], MF.shape[0], FT.shape[0], FF.shape[0]))
    print()

    for i in range(FF.shape[0]):
        list_ids.append(FF[i])
        labels[FF[i]] = torch.Tensor([0, 0])
    for i in range(FT.shape[0]):
        list_ids.append(FT[i])
        labels[FT[i]] = torch.Tensor([1, 0])
    for i in range(MF.shape[0]):
        list_ids.append(MF[i])
        labels[MF[i]] = torch.Tensor([0, 1])
    for i in range(MT.shape[0]):
        list_ids.append(MT[i])
        labels[MT[i]] = torch.Tensor([1, 1])

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

def create_dataset_actual_fivecrop(path, attribute, protected_attribute, params, augment, dataset, number=0, split='train'):

    img_path = path+'/img_align_celeba/'
    split_path = path+'/list_eval_partition_celeba.txt'
    attr_path = path+'/list_attr_celeba.txt'
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
            T.FiveCrop(224),
            #Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
            #normalize
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(t) for t in tensors]))
        ])

    #print(transform)
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader

def create_dataset_one_prot_only(path, attribute, protected_attribute, params, augment, dataset, protected_val = 0, number=0, split='train'):

    img_path = path+'/img_align_celeba/'
    split_path = path+'/list_eval_partition_celeba.txt'
    attr_path = path+'/list_attr_celeba.txt'
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
        if int((int(temp[protected_attribute+1])+1)/2)==protected_val:
            list_ids.append(img_path+temp[0])
            attr[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

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

    #print(transform)
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader

def create_dataset_single_domain(path, label_path, range1, params, dataset, augment):

    list_ids = []
    labeldata = pickle.load(open(label_path, 'rb'))
    labels = {}
    for i in range(range1[0], range1[1]):
        list_ids.append(path + 'gen_'+str(i)+'.jpg')
        labels[path+ 'gen_'+str(i)+'.jpg'] = labeldata[i]

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

    dset = dataset(list_ids, labels)
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

