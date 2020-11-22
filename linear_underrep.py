import numpy as np
from sklearn import svm
import pickle
import argparse
import parse_args
from utils import make_dir

if __name__=="__main__":
    
    opt = parse_args.collect_args_linear()
    np.random.seed(opt['random_seed'])
    
    X = pickle.load(open('record/GAN_model/latent_vectors_extra.pkl', 'rb')) 
    g = pickle.load(open('data/extra_scores/'+opt['prot_attr_name']+'_scores.pkl', 'rb')) 
    a = pickle.load(open('data/extra_scores/'+opt['attr_name']+'_scores.pkl', 'rb'))
    thresh_g = pickle.load(open('record/baseline/{}/test_results.pkl'.format(opt['prot_attr_name']), 'rb'))['threshold']
    g = np.where(g>thresh_g, 1, 0)
    thresh_a = pickle.load(open('record/baseline/{}/test_results.pkl'.format(opt['attr_name']), 'rb'))['threshold']
    a = np.where(a>thresh_a, 1, 0)
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
    
    #for a in idx:
    #    print(len(a))
    #spec = int(opt['protected_val']*2 + opt['attr_val'])
    #print(idx)
    idx_all = []
    other_num = (12000 - 2*opt['number'])//2
    #print(other_num)
    for i in range(2):
        if i != opt['attr_val']:
            idx_all.append(idx[i][:other_num])
            idx_all.append(idx[2+i][:other_num])
        else:
            idx_all.append(idx[i][:opt['number']])
            idx_all.append(idx[2+i][:opt['number']])
    
    idx_all = np.concatenate(idx_all).astype(int).tolist()
    #print(max(idx_all))
    #print(len(idx_all))
    
    
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
    #X_on_hyperplane = np.zeros((160000, X.shape[1]))
    #X_perp = np.zeros((160000, X.shape[1]))
    #X_perp_on_hyperplane = np.zeros((160000, X.shape[1]))
    
    X_orig = pickle.load(open('record/GAN_model/latent_vectors.pkl', 'rb'))
    
    a_orig = pickle.load(open('data/fake_images/all_'+opt['attr_name']+'_scores.pkl', 'rb'))
    
    print(opt['number'], clf_a.score(X_orig[:15000], a_orig[:15000]))

    """
    for j in range(15000, 175000):
        x = X_orig[j]
        dist = np.sum(clf_g.coef_ * x) + clf_g.intercept_
        
        X_all[j-15000] = x - ((2*dist)/sin_theta)* g_perp_a
        #X_on_hyperplane[j-15000] = x - ((dist)/sin_theta)* g_perp_a
        #X_perp[j-15000] = x - ((2*dist)*clf_g.coef_)
        #X_perp_on_hyperplane[j-15000] = x - ((dist)*clf_g.coef_)
        

    
    #with open('data/fake_images/'+opt['attr_name']+'_scores.pkl', 'wb+') as handle:
    #    pickle.dump(a[15000:175000], handle)
    
    orig_save = 'record/GAN_model/underrep/'
    #if opt['protected_attribute']!=20:
    #    orig_save+='protected'+opt['prot_attr_name']+'/'
    make_dir(orig_save)
    orig_save = 'record/GAN_model/underrep/{}'.format(opt['attr_name'])
    make_dir(orig_save)
    #make_dir(orig_save+'perp')
    #make_dir(orig_save+'perp_on_hyperplane')
    #make_dir(orig_save+'on_hyperplane')
    
    with open(orig_save+'/latent_vectors_'+str(opt['number'])+'_'+str(opt['attr_val'])+'.pkl', 'wb+') as handle:
        pickle.dump(X_all, handle)
    
    #with open(orig_save+'perp/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
    #    pickle.dump(X_perp, handle)

    #with open(orig_save+'on_hyperplane/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
    #    pickle.dump(X_on_hyperplane, handle)
    
    #with open(orig_save+'perp_on_hyperplane/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
    #    pickle.dump(X_perp_on_hyperplane, handle)

    """
