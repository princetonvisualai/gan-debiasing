import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score
from os import listdir, path, mkdir
from scipy.ndimage import gaussian_filter


def compute_weight(domain, target):

    cp = target.sum()
    cn = target.shape[0]-cp
    cn_dn = ((1-target)*(1-domain)).sum()
    cn_dp = ((1-target)*(domain)).sum()
    cp_dn = ((target)*(1-domain)).sum()
    cp_dp = ((target)*(domain)).sum()

    weights = (target*cp + (1-target)*cn) / (2*(
                    (1-target)*(1-domain)*cn_dn
                    + (1-target)*domain*cn_dp
                    + target*(1-domain)*cp_dn
                    + target*domain*cp_dp
                   )
                )


    return weights

def compute_class_weight(loader, device, dtype):

    cp = 0
    cn = 0
    cn_dn = 0
    cn_dp = 0
    cp_dn = 0
    cp_dp = 0

    weights = []
    for x,y in loader:
        y = y.to(device=device, dtype=dtype)

        class_label = y[:,0]
        domain_label = y[:,1]
        cp += class_label.sum() # class is positive
        cn += (y.shape[0] - class_label.sum() )# class is negative
        cn_dn += ((class_label + domain_label)==0).sum() # class is negative, domain is negative
        cn_dp += ((class_label - domain_label)==-1).sum()
        cp_dn += ((class_label - domain_label)==1).sum()
        cp_dp += ((class_label + domain_label)==2).sum()
    for x,y in loader:
        y = y.to(device=device, dtype=dtype)
        class_label = y[:, 0]
        domain_label = y[:, 1]

        weights.append(
            (class_label*cp + (1-class_label)*cn) /
                (2*(
                    (1-class_label)*(1-domain_label)*cn_dn
                    + (1-class_label)*domain_label*cn_dp
                    + class_label*(1-domain_label)*cp_dn
                    + class_label*domain_label*cp_dp
                   )
                )
        )


    weights = torch.cat(weights)
    return weights

def compute_class_weight_multi(loader, device, dtype):

    cp = 0
    cn = 0
    cn_dnn = 0
    cn_dpn = 0
    cp_dnn = 0
    cp_dpn = 0
    cn_dnp = 0
    cn_dpp = 0
    cp_dnp = 0
    cp_dpp = 0

    weights = []
    for x,y in loader:
        y = y.to(device=device, dtype=dtype)

        class_label = y[:,0]
        domain_label = y[:,1]
        cp += class_label.sum() # class is positive
        cn += (y.shape[0] - class_label.sum() )# class is negative
        cn_dnn+= ((1-class_label)*(torch.where(domain_label==0, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()
        cn_dnp+= ((1-class_label)*(torch.where(domain_label==1, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()
        cn_dpn+= ((1-class_label)*(torch.where(domain_label==2, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()
        cn_dpp+= ((1-class_label)*(torch.where(domain_label==3, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()

        cp_dnn+= ((1-class_label)*(torch.where(domain_label==0, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()
        cp_dnp+= ((1-class_label)*(torch.where(domain_label==1, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()
        cp_dpn+= ((1-class_label)*(torch.where(domain_label==2, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()
        cp_dpp+= ((1-class_label)*(torch.where(domain_label==3, torch.ones_like(domain_label), torch.zeros_like(domain_label)))).sum()


    for x,y in loader:
        y = y.to(device=device, dtype=dtype)
        class_label = y[:, 0]
        domain_label = y[:, 1]
        domain_1 = domain_label//2
        domain_2 = domain_label%2

        weights.append(
            (class_label*cp + (1-class_label)*cn) /
                (2*(
                    (1-class_label)*(1-domain_1)*(1-domain_2)*cn_dnn
                    + (1-class_label)*(1-domain_1)*(domain_2)*cn_dnp
                    + (1-class_label)*domain_1*(1-domain_2)*cn_dpn
                    + (1-class_label)*domain_1*(domain_2)*cn_dpp
                    + class_label*(1-domain_1)*(1-domain_2)*cp_dnn
                    + class_label*(1-domain_1)*(domain_2)*cp_dnp
                    + class_label*domain_1*(1-domain_2)*cp_dpn
                    + class_label*domain_1*(domain_2)*cp_dpp
                   )
                )
        )


    weights = torch.cat(weights)
    return weights



def get_all_attr():

    return ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs',  'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

def get_attr_list():
    return [1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 17, 19, 21, 23, 25, 26, 27, 28, 31, 32, 33, 34, 35, 39]


def make_dir(pathname):
    if not path.isdir(pathname):
        mkdir(pathname)


def get_threshold(targets_all, scores_all):
    best_t = -1.0
    best_acc = 0.0
    for t in range(1,10):
        thresh = 0.1*t
        curr_scores = np.where(scores_all>thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        #print(thresh, acc, best_acc, flush=True)
        if acc>best_acc:
            best_acc = acc
            best_t = thresh
    one_dec = best_t

    for t in range(1,20):
        thresh =(one_dec-0.1) + 0.01*t
        curr_scores = np.where(scores_all>thresh, 1, 0)
        acc = f1_score(targets_all, curr_scores)
        #print(thresh, acc, best_acc, flush=True)
        if acc>best_acc:
            best_acc = acc
            best_t = thresh

    return best_acc, best_t

def calibrated_threshold(targets, scores):
    cp = int(targets.sum())
    scores_copy = np.copy(scores)
    scores_copy.sort()
    #print(cp)
    thresh = scores_copy[-cp]
    return thresh

def kl(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def smoothed_hist_kl_distance(a, b, nbins=50, sigma=1):
    ahist = np.histogram(a, bins=nbins)[0]
    bhist = np.histogram(b, bins=nbins)[0]

    asmooth = gaussian_filter(ahist, sigma)
    bsmooth = gaussian_filter(bhist, sigma)

    asmooth = asmooth/asmooth.sum() + 1e-6
    bsmooth = bsmooth/bsmooth.sum() + 1e-6

    return kl(asmooth, bsmooth), kl(bsmooth, asmooth)

def bootstrap_deo(domain, targets, pred, repeat=500):
    max_val = targets.squeeze().shape[0]
    deo = np.zeros(repeat)
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        targets_i = targets[rand_index]
        pred_i = pred[rand_index]
        domain_i = domain[rand_index]

        g0 = np.argwhere(domain_i==0)
        g1 = np.argwhere(domain_i==1)

        deo[i] = np.abs((1-recall_score(targets_i[g0], pred_i[g0]))-(1-recall_score(targets_i[g1],pred_i[g1])))


    return np.median(deo), np.std(deo)


def bootstrap_ap(targets_all, scores_all,  repeat=500):
    max_val = targets_all.squeeze().shape[0]
    avg_prec_weights = np.zeros(repeat)
    avg_prec = np.zeros(repeat)
    #print(targets_all[:10], scores_all.shape)
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        targets = targets_all[rand_index]
        scores = scores_all[rand_index]

        avg_prec[i] = average_precision_score(targets, scores)


    return np.median(avg_prec), np.std(avg_prec)

def bog_attribute_to_task(bog_tilde, bog_gt_g, bog_tilde_train=None, toprint=True, disaggregate=False, num_attributes=None, total_images=None, num_attributes_train=None, total_images_train=None):
    if num_attributes is None: # need to be provided if multi-label, this applies to bog_gt_g counts
        num_attributes = np.sum(bog_tilde, axis=0)
    if total_images is None: # need to be provided if attribute is not mutually exclusive
        total_images = np.sum(num_attributes)
    if bog_tilde_train is None:
        bog_tilde_train = bog_tilde
    if num_attributes_train is None:
        num_attributes_train = np.sum(bog_tilde_train, axis=0)
    if total_images_train is None:
        total_images_train = np.sum(num_attributes_train)

    data_bog = np.zeros_like(bog_tilde)
    data_bog = bog_tilde / np.expand_dims(num_attributes, 0)

    pred_bog = np.zeros_like(bog_gt_g)
    pred_bog = bog_gt_g / np.expand_dims(num_attributes, 0)

    #p_a_t = np.zeros_like(data_bog)
    #for i in range(len(data_bog)):
    #    p_a_t[i] = bog_tilde[i]/np.sum(bog_tilde[i])
    #p_a = num_attributes/np.sum(num_attributes)

    p_t_a = np.zeros_like(data_bog)
    p_t_a = bog_tilde_train / np.expand_dims(num_attributes_train, 0)
    p_t = np.sum(bog_tilde_train, axis=1)/total_images_train

    diff = np.zeros_like(data_bog)
    for i in range(len(data_bog)):
        for j in range(len(data_bog[0])):
            diff[i][j] = pred_bog[i][j] - data_bog[i][j]
            #indicator = np.sign(p_a_t[i][j] - p_a[j])
            indicator = np.sign(p_t_a[i][j] - p_t[i]) # original one
            if indicator == 0:
                diff[i][j] = 0
            elif indicator == -1:
                diff[i][j] = - diff[i][j]
    if disaggregate:
        diff_before = diff.copy()
    value = np.nanmean(diff)
    if toprint:
        print("Attribute->Task: {}".format(value))
    if disaggregate:
        return diff_before, value
    return value

def get_at(running_labels, running_preds):
    bog_tilde = np.zeros((2, 2))
    bog_gt_g = np.zeros((2, 2))
    gt_female = np.where(running_labels[:, 1] == 0)[0]
    gt_male = np.where(running_labels[:, 1] == 1)[0]
    gt_kitchen = np.where(running_labels[:, 0] == 0)[0]
    gt_sports = np.where(running_labels[:, 0] == 1)[0]
    for i, objs in enumerate([running_labels, running_preds]):
        female = np.where(objs[:, 1] == 0)[0]
        male = np.where(objs[:, 1] == 1)[0]
        kitchen = np.where(objs[:, 0] == 0)[0]
        sports = np.where(objs[:, 0] == 1)[0]
        if i == 0:
            bog_tilde[0][0] = len(set(kitchen)&set(female))
            bog_tilde[0][1] = len(set(kitchen)&set(male))
            bog_tilde[1][0] = len(set(sports)&set(female))
            bog_tilde[1][1] = len(set(sports)&set(male))
        elif i == 1:
            bog_gt_g[0][0] = len(set(kitchen)&set(gt_female))
            bog_gt_g[0][1] = len(set(kitchen)&set(gt_male))
            bog_gt_g[1][0] = len(set(sports)&set(gt_female))
            bog_gt_g[1][1] = len(set(sports)&set(gt_male))
    at = bog_attribute_to_task(bog_tilde, bog_gt_g, toprint=False)
    return at


def bootstrap_bias_amp(domain, targets, pred, repeat=500):

    test_labels = np.zeros((targets.shape[0], 2))
    test_labels[:, 0] = targets
    test_labels[:, 1] = domain
    max_val = targets.shape[0]
    repeat = 500
    test_pred = np.zeros((targets.shape[0], 2))
    test_pred[:,0]= pred
    test_pred[:, 1] = domain
    auc_bias = []

    max_val = targets.shape[0]
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        labels_i = test_labels[rand_index]
        pred_i = test_pred[rand_index]
        auc_bias.append(get_at(labels_i, pred_i))

    return np.median(auc_bias), np.std(auc_bias)

def bootstrap_kl(domain_all, targets_all, scores_all, repeat=500):
    max_val = targets_all.shape[0]
    avg_prec = np.zeros(repeat)

    a_b_sublist = []
    b_a_sublist = []
    a_b_pos_sublist = []
    b_a_pos_sublist = []
    a_b_neg_sublist = []
    b_a_neg_sublist = []
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        targets = targets_all[rand_index]
        scores = scores_all[rand_index]
        domain = domain_all[rand_index]

        MT = np.logical_and(domain==1, targets==1)
        MF = np.logical_and(domain==1, targets==0)
        FT = np.logical_and(domain==0, targets==1)
        FF = np.logical_and(domain==0, targets==0)

        nbin = 50 # Number of histogram bins

        a_b, b_a = smoothed_hist_kl_distance(scores[domain==0], scores[domain==1], nbins=nbin)
        a_b_sublist.append(a_b); b_a_sublist.append(b_a)

        a_b_pos, b_a_pos = smoothed_hist_kl_distance(scores[np.logical_and(domain==0, targets==1)],
            scores[np.logical_and(domain==1, targets==1)], nbins=nbin)
        a_b_pos_sublist.append(a_b_pos); b_a_pos_sublist.append(b_a_pos)

        a_b_neg, b_a_neg = smoothed_hist_kl_distance(scores[np.logical_and(domain==0, targets==0)],
            scores[np.logical_and(domain==1, targets==0)], nbins=nbin)
        a_b_neg_sublist.append(a_b_neg); b_a_neg_sublist.append(b_a_neg)

    #a_b_pos_list.append(a_b_pos_sublist)
    #b_a_pos_list.append(b_a_pos_sublist)
    #a_b_neg_list.append(a_b_neg_sublist)
    #b_a_neg_list.append(b_a_neg_sublist)

    return np.median(a_b_pos_sublist+a_b_neg_sublist+b_a_pos_sublist+b_a_neg_sublist), np.std(a_b_pos_sublist+a_b_neg_sublist+b_a_pos_sublist+b_a_neg_sublist)
