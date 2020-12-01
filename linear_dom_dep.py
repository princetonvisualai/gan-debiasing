import numpy as np
from sklearn import svm
import pickle
import utils
import parse_args

# Code to compute latent vectors for extension of model that uses different target attribute hyperplanes 
# for vectors of different protected attribute values. 
# manipulated vectors are stored in 'record/GAN_model/domain_dependent/latent_vectors_{attr_name}.pkl'


def optimize_z(A, lr, z):
    
    z_prime = np.random.normal(z.shape[0])
    
    u, s, vh = np.linalg.svd(A)
    change = 100.0

    while change>1.0:
        
        z_prime_project = z_prime - (np.sum(z_prime*u[:, 0])*u[:, 0] + np.sum(z_prime*u[:, 1])*u[:, 1])
        loss = np.sum((z_prime_project-z)*(z_prime_project - z))
        dloss = 2*(z_prime_project-z)
        z_prime_new = z-lr*dloss

        change = np.linalg.norm(z_prime_new - z_prime)
        z_prime = z_prime_new
    return z_prime_project[:-1]/(z_prime_project[-1])


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
    
    clf_g = svm.LinearSVC(max_iter=500000) 
    clf_g.fit(X_train, g_train)

    clf_g_norm = np.linalg.norm(clf_g.coef_)
    clf_g.coef_ = clf_g.coef_/(clf_g_norm)
    clf_g.intercept_ = clf_g.intercept_/clf_g_norm

    g0 = list(np.argwhere(g_train==0).squeeze())
    g1 = list(np.argwhere(g_train==1).squeeze())
    

    clf_a0 = svm.LinearSVC(max_iter=500000) 
    clf_a0.fit(X_train[g0], a_train[g0])

    clf_a0_norm = np.linalg.norm(clf_a0.coef_)
    clf_a0.coef_ = clf_a0.coef_/(clf_a0_norm)
    clf_a0.intercept_ = clf_a0.intercept_/clf_a0_norm

    clf_a1 = svm.LinearSVC(max_iter=500000) 
    clf_a1.fit(X_train[g1], a_train[g1])

    clf_a1_norm = np.linalg.norm(clf_a1.coef_)
    clf_a1.coef_ = clf_a1.coef_/(clf_a1_norm)
    clf_a1.intercept_ = clf_a1.intercept_/clf_a1_norm
    
    
    X_all = np.zeros((160000, X.shape[1]))
    
    for i in range(15000, 175000):
        
        z = np.zeros(X.shape[1]+1)
        z[:-1] = X[i]
        z[-1] = 1

        dist_g = np.sum(clf_g.coef_ * X[i]) + clf_g.intercept_
        
        A = np.zeros((X.shape[1]+1, 2))
        A[:-1, 0] = clf_g.coef_
        A[-1, 0] = clf_g.intercept_ + dist_g

        if g[i] == 0:
            dist_a = np.sum(clf_a0.coef_ * X[i]) + clf_a0.intercept_
            A[:-1, 1] = clf_a1.coef_ 
            A[-1, 1] = clf_a1.intercept_ - dist_a 

        elif g[i] == 1:
            dist_a = np.sum(clf_a1.coef_ * X[i]) + clf_a1.intercept_
            A[:-1, 1] = clf_a0.coef_ 
            A[-1, 1] = clf_a0.intercept_ - dist_a 
        
        X_all[i-15000] = optimize_z(A, 0.01, z)
        #print(np.sum(clf_g.coef_ * X_all[i-15000]) + clf_g.intercept_, dist_g)
        #print(np.sum(clf_a0.coef_ * X_all[i-15000]) + clf_a0.intercept_, dist_a)
        #print(np.sum(clf_a1.coef_ * X_all[i-15000]) + clf_a1.intercept_, dist_a)
   
        
    utils.make_dir('record/GAN_model/domain_dependent/')
    
    with open('record/GAN_model/domain_dependent/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
        pickle.dump(X_all, handle)
    
