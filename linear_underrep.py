import numpy as np
from sklearn import svm
import pickle
import argparse
import parse_args
from utils import make_dir

# Target attribute hyperplanes are computed by using different fractions of positive and negative vectors.  
# Manipulated vectors are stored in 'record/GAN_model/underrep/latent_vectors_{attr_name}_{number}_{val}.pkl',
# where 2*{number} of vectors labelled with {val} are used to compute the hyperplane

if __name__=="__main__":
    
    opt = parse_args.collect_args_linear()
    np.random.seed(opt['random_seed'])
    
    X = pickle.load(open('record/GAN_model/latent_vectors_extra.pkl', 'rb')) 
    g = pickle.load(open('data/extra_scores/'+opt['prot_attr_name']+'_scores.pkl', 'rb')) 
    a = pickle.load(open('data/extra_scores/'+opt['attr_name']+'_scores.pkl', 'rb'))
    print(a.sum())
    print(opt)
    
    idx = [[], [], [], []]
    
    for i in range(g.shape[0]):
        if g[i]==0 and a[i]==0:
            idx[0].append(i)
        if g[i]==0 and a[i]==1:
            idx[1].append(i)
        if g[i]==1 and a[i]==0:
            idx[2].append(i)
        elif g[i]==1 and a[i]==1:
            idx[3].append(i)
    
    idx_all = []
    other_num = (12000 - 2*opt['number'])//2
    for i in range(2):
        if i != opt['attr_val']:
            idx_all.append(idx[i][:other_num])
            idx_all.append(idx[2+i][:other_num])
        else:
            idx_all.append(idx[i][:opt['number']])
            idx_all.append(idx[2+i][:opt['number']])
    
    idx_all = np.concatenate(idx_all).astype(int).tolist()
    
    
    print(a.shape, g.shape) 
    X_train = np.array([X[i] for i in idx_all])
    a_train = np.array([a[i] for i in idx_all])
    g_train = np.array([g[i] for i in idx_all])
    
    clf_g = svm.LinearSVC(max_iter=500000) 
    clf_g.fit(X_train, g_train)

    clf_g_norm = np.linalg.norm(clf_g.coef_)
    clf_g.coef_ = clf_g.coef_/(clf_g_norm)
    clf_g.intercept_ = clf_g.intercept_/clf_g_norm

    clf_a = svm.LinearSVC(max_iter=500000) 
    clf_a.fit(X_train, a_train)

    clf_a_norm = np.linalg.norm(clf_a.coef_)
    clf_a.coef_ = clf_a.coef_/(clf_a_norm)
    clf_a.intercept_ = clf_a.intercept_/clf_a_norm

    g_perp_a = clf_g.coef_ - (np.sum(clf_g.coef_* clf_a.coef_))*clf_a.coef_

    g_perp_a = g_perp_a/np.linalg.norm(g_perp_a)

    cos_theta = np.sum(clf_g.coef_*clf_a.coef_)
    sin_theta = np.sqrt(1 - cos_theta*cos_theta)

    X_all = np.zeros((160000, X.shape[1]))
    
    X_orig = pickle.load(open('record/GAN_model/latent_vectors.pkl', 'rb'))
    
    a_orig = pickle.load(open('data/fake_images/all_'+opt['attr_name']+'_scores.pkl', 'rb'))
    

    
    for j in range(15000, 175000):
        x = X_orig[j]
        dist = np.sum(clf_g.coef_ * x) + clf_g.intercept_
        
        X_all[j-15000] = x - ((2*dist)/sin_theta)* g_perp_a
    
    orig_save = 'record/GAN_model/underrep/'
    make_dir(orig_save)
    orig_save = 'record/GAN_model/underrep/{}'.format(opt['attr_name'])
    make_dir(orig_save)
    
    with open(orig_save+'/latent_vectors_'+str(opt['number'])+'_'+str(opt['attr_val'])+'.pkl', 'wb+') as handle:
        pickle.dump(X_all, handle)
    
