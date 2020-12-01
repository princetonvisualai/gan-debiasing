import numpy as np
from sklearn import svm
import pickle
import argparse
import parse_args
from utils import make_dir

# Code to compute latent vectors for paired sampling, as well as all ablations.  
# Manipulated vectors are stored in 'record/GAN_model/latent_vectors_{attr_name}.pkl'

if __name__=="__main__":
    
    opt = parse_args.collect_args_linear()
    np.random.seed(opt['random_seed'])
    
    X = pickle.load(open('record/GAN_model/latent_vectors.pkl', 'rb')) 
    g = pickle.load(open('data/fake_images/all_'+opt['prot_attr_name']+'_scores.pkl', 'rb')) 
    a = pickle.load(open('data/fake_images/all_'+opt['attr_name']+'_scores.pkl', 'rb'))

    X_train = X[:10000, :]
    g_train = g[:10000]
    a_train = a[:10000]
    
    X_val = X[10000:15000, :]
    g_val = g[10000:15000]
    a_val = a[10000:15000]
    print(a_val.sum())
    print(opt)
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
    X_on_hyperplane = np.zeros((160000, X.shape[1]))
    X_perp = np.zeros((160000, X.shape[1]))
    X_perp_on_hyperplane = np.zeros((160000, X.shape[1]))

    for j in range(15000, 175000):
        x = X[j]
        dist = np.sum(clf_g.coef_ * x) + clf_g.intercept_
        
        X_all[j-15000] = x - ((2*dist)/sin_theta)* g_perp_a
        X_on_hyperplane[j-15000] = x - ((dist)/sin_theta)* g_perp_a
        X_perp[j-15000] = x - ((2*dist)*clf_g.coef_)
        X_perp_on_hyperplane[j-15000] = x - ((dist)*clf_g.coef_)
        
    orig_save = 'record/GAN_model/'
    make_dir(orig_save)  
    make_dir(orig_save+'perp')
    make_dir(orig_save+'perp_on_hyperplane')
    make_dir(orig_save+'on_hyperplane')
   
    with open(orig_save+'latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
        pickle.dump(X_all, handle)

    with open(orig_save+'perp/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
        pickle.dump(X_perp, handle)

    with open(orig_save+'on_hyperplane/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
        pickle.dump(X_on_hyperplane, handle)
    
    with open(orig_save+'perp_on_hyperplane/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
        pickle.dump(X_perp_on_hyperplane, handle)

    
