import pickle
import argparse
import numpy as np
import pandas as pd
import time
from sklearn import svm

import parse_args
from utils import make_dir

if __name__=="__main__":

    opt = parse_args.collect_args_linear()
    np.random.seed(opt['random_seed'])

    # Load latent vectors
    z_inv_val = pickle.load(open('zinv/inverted_latent_vectors_val.pkl', 'rb'))
    z_ran = pickle.load(open('zinv/random_latent_vectors.pkl', 'rb'))

    # Load labels of the latent vectors
    attr_path = 'data/celeba/list_attr_celeba.txt'
    with open(attr_path, "r") as f:
        attr = pd.read_csv(f, delim_whitespace=True, header=1)
    g_inv = (attr['Male'][:162770].values == 1).astype(np.float16)
    a_inv = (attr[opt['attr_name']][:162770].values == 1).astype(np.float16)
    g_ran = pickle.load(open('random_scores/all_Male_scores.pkl', 'rb'))
    a_ran = pickle.load(open('random_scores/all_{}_scores.pkl'.format(opt['attr_name']), 'rb'))

    # Estimate hyperplanes
    z_train = np.concatenate((z_inv_val, z_ran[:15000]))
    g_train = np.concatenate(((attr['Male'][162770:182637].values == 1).astype(np.float16), g_ran[:15000]))
    a_train = np.concatenate(((attr[opt['attr_name']][162770:182637].values == 1).astype(np.float16), a_ran[:15000]))

    np.random.seed(1)
    selected = np.random.choice(np.arange(z_train.shape[0]), size=15000, replace=False)
    z_train = z_train[selected]
    a_train = a_train[selected]
    g_train = g_train[selected]
    print('Hyperplane estimated with {} samples'.format(z_train.shape[0]))

    ha = svm.LinearSVC(max_iter=500000)
    ha.fit(z_train, a_train)
    ha_norm = np.linalg.norm(ha.coef_)
    ha.coef_ = ha.coef_/ha_norm # w_a
    ha.intercept_ = ha.intercept_/ha_norm # b_a
    wa = ha.coef_
    ba = ha.intercept_

    hg = svm.LinearSVC(max_iter=500000)
    hg.fit(z_train, g_train)
    hg_norm = np.linalg.norm(hg.coef_)
    hg.coef_ = hg.coef_/hg_norm # w_g
    hg.intercept_ = hg.intercept_/hg_norm # b_g
    wg = hg.coef_
    bg = hg.intercept_

    # Compute z'
    g_perp_a = wg - np.sum(wg*wa)*wa # w_g - (w_g \cdot w_a)*w_a
    g_perp_a = g_perp_a/np.linalg.norm(g_perp_a) # (w_g - (w_g \cdot w_a)*w_a) / sin_theta
    a_perp_g = wa - np.sum(wa*wg)*wg # w_a - (w_a \cdot w_g)*w_g
    a_perp_g = a_perp_g/np.linalg.norm(a_perp_g) # (w_a - (w_a \cdot w_g)*w_g) / sin_theta
    sin_theta = np.sqrt(1 - np.sum(wg*wa)**2) # sqrt(1 - (w_g \cdot w_a)^2)

    z_prime = np.zeros((162770, 512))
    for j in range(162770):
        dist = np.sum(wg * z[j]) + bg # w_g \cdot z + b_g
        z_prime[j] = z[j] - ((2*dist)/sin_theta) * g_perp_a # z'

    # Save new latent vectors
    make_dir('record/inverted_raninv/prime')
    with open('record/inverted_raninv/prime/latent_vectors_{}.pkl'.format(opt['attr_name']), 'wb+') as handle:
        pickle.dump(z_prime, handle)
