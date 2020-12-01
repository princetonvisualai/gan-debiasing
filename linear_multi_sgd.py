import numpy as np
from sklearn import svm
import pickle
import utils
import parse_args


# Code to compute latent vectors for comparison with Sharmanska et al. 
# manipulated vectors are stored in 'record/GAN_model/multi_setting/latent_vectors_{attr_name}.pkl'
# Vectors 0-159,999 flip gender expression scores, while maintaining young scores.
# Vectors 160,000-319,999 flip young scores, while maintaining gender expression scores.
# Vectors 320,000-479,999 flip both. 

def optimize_z(A, lr, z):
    
    z_prime = np.random.normal(z.shape[0])
    
    u, s, vh = np.linalg.svd(A)
    #print(s, u.shape, vh.shape)
    change = 100.0

    while change>1.0:
        
        z_prime_project = z_prime - (np.sum(z_prime*u[:, 0])*u[:, 0] + np.sum(z_prime*u[:, 1])*u[:, 1] + np.sum(z_prime*u[:, 2])*u[:, 2])
        loss = np.sum((z_prime_project-z)*(z_prime_project - z))
        dloss = 2*(z_prime_project-z)
        z_prime_new = z-lr*dloss
        #print(np.dot(z_prime_project,A))

        change = np.linalg.norm(z_prime_new - z_prime)
        z_prime = z_prime_new
        #print(change)
    return z_prime_project[:-1]/(z_prime_project[-1])


if __name__=="__main__":
    
    opt = parse_args.collect_args_linear()
    np.random.seed(opt['random_seed'])
    
    X = pickle.load(open('record/GAN_model/latent_vectors.pkl', 'rb')) 
    g1 = pickle.load(open('data/fake_images/all_Male_scores.pkl', 'rb')) 
    g2 = pickle.load(open('data/fake_images/all_Young_scores.pkl', 'rb')) 
    a = pickle.load(open('data/fake_images/all_'+opt['attr_name']+'_scores.pkl', 'rb'))

    X_train = X[:10000, :]
    g1_train = g1[:10000]
    g2_train = g2[:10000]
    a_train = a[:10000]
    
    X_val = X[10000:15000, :]
    g1_val = g1[10000:15000]
    g2_val = g2[10000:15000]
    a_val = a[10000:15000]
    
    clf_g1 = svm.LinearSVC(max_iter=500000) 
    clf_g1.fit(X_train, g1_train)

    clf_g1_norm = np.linalg.norm(clf_g1.coef_)
    clf_g1.coef_ = clf_g1.coef_/(clf_g1_norm)
    clf_g1.intercept_ = clf_g1.intercept_/clf_g1_norm

    clf_g2 = svm.LinearSVC(max_iter=500000) 
    clf_g2.fit(X_train, g2_train)

    clf_g2_norm = np.linalg.norm(clf_g2.coef_)
    clf_g2.coef_ = clf_g2.coef_/(clf_g2_norm)
    clf_g2.intercept_ = clf_g2.intercept_/clf_g2_norm
    
    clf_a = svm.LinearSVC(max_iter=500000) 
    clf_a.fit(X_train, a_train)

    clf_a_norm = np.linalg.norm(clf_a.coef_)
    clf_a.coef_ = clf_a.coef_/(clf_a_norm)
    clf_a.intercept_ = clf_a.intercept_/clf_a_norm
    
    
    X_all = np.zeros((480000, X.shape[1]))
    #First 160000 flips g1 = gender expression
    #Next 160000 flips g2 = young
    #Last 160000 flips both
    
    for i in range(160000):
        
            
        z = np.zeros(X.shape[1]+1)
        z[:-1] = X[i+15000]
        z[-1] = 1

        dist_g1 = np.sum(clf_g1.coef_ * X[i+15000]) + clf_g1.intercept_
        dist_g2 = np.sum(clf_g2.coef_ * X[i+15000]) + clf_g2.intercept_
        dist_a = np.sum(clf_a.coef_ * X[i+15000]) + clf_a.intercept_

        A = np.zeros((3, X.shape[1]+1, 3))
        
        A[0,:-1, 0] = clf_g1.coef_
        A[0,-1, 0] = clf_g1.intercept_ + dist_g1

        A[0, :-1, 1] = clf_g2.coef_
        A[0, -1, 1] = clf_g2.intercept_ - dist_g2

        A[0, :-1, 2] = clf_a.coef_
        A[0, -1, 2] = clf_a.intercept_ - dist_a

        
        A[1,:-1, 0] = clf_g1.coef_
        A[1, -1, 0] = clf_g1.intercept_ - dist_g1

        A[1, :-1, 1] = clf_g2.coef_
        A[1, -1, 1] = clf_g2.intercept_ + dist_g2

        A[1, :-1, 2] = clf_a.coef_
        A[1, -1, 2] = clf_a.intercept_ - dist_a
        
        
        A[2,:-1, 0] = clf_g1.coef_
        A[2,-1, 0] = clf_g1.intercept_ + dist_g1

        A[2, :-1, 1] = clf_g2.coef_
        A[2, -1, 1] = clf_g2.intercept_ + dist_g2

        A[2, :-1, 2] = clf_a.coef_
        A[2, -1, 2] = clf_a.intercept_ - dist_a
        
        
        X_all[i] = optimize_z(A[0], 0.01, z)
        X_all[160000+i] = optimize_z(A[1], 0.01, z)
        X_all[320000+i] = optimize_z(A[2], 0.01, z)

    
        
    utils.make_dir('record/GAN_model/multi_setting')
    
    with open('record/GAN_model/multi_setting/latent_vectors_'+opt['attr_name']+'.pkl', 'wb+') as handle:
        pickle.dump(X_all, handle)
    
