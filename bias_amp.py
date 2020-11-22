import numpy as np
from scipy import stats
import pickle
import utils
from sklearn import metrics
from os import path

def bog_mals(bog_tilde, bog_pred, bog_gt_g=None, bog_gt_o=None, bog_tilde_train=None, toprint=True):
    if bog_tilde_train is None:
        bog_tilde_train = bog_tilde
    data_bog = bog_tilde / np.sum(bog_tilde, axis=1, keepdims=True)
    pred_bog = bog_pred / np.sum(bog_pred, axis=1, keepdims=True)
    diff = np.zeros_like(data_bog)
    for i in range(len(data_bog)):
        for j in range(len(data_bog[0])):
            if data_bog[i][j] > (1./len(data_bog[0])):
                diff[i][j] = pred_bog[i][j] - data_bog[i][j]
    value = (1./len(data_bog))*(np.nansum(diff))
    #value = (1./len(data_bog))*(np.sum(diff))
    if toprint:
        print("BOG men also like shopping: {}".format(value))

    if bog_gt_g is not None and bog_gt_o is not None:
        g_to_o_component = bog_mals(bog_tilde, bog_gt_g, bog_tilde_train=bog_tilde_train, toprint=False)
        o_to_g_component = bog_mals(bog_tilde, bog_gt_o, bog_tilde_train=bog_tilde_train, toprint=False)
        print("Components: g->o: {0}, o->g: {1}".format(g_to_o_component, o_to_g_component))
        return value, g_to_o_component, o_to_g_component
    return value

# shape of bogs is |O| x |G|
def bog_task_to_attribute(bog_tilde, bog_gt_o, bog_tilde_train=None, toprint=True, disaggregate=False, num_attributes=None, total_images=None, num_attributes_train=None, total_images_train=None):
    if num_attributes is None: # need to be provided if multi-label
        num_attributes = np.sum(bog_tilde, axis=0)
    if total_images is None: # need to be provided if attribute is not mutually exclusive
        total_images = np.sum(num_attributes)
    if bog_tilde_train is None:
        bog_tilde_train = bog_tilde
    if num_attributes_train is None:
        num_attributes_train = np.sum(bog_tilde_train, axis=0)
    if total_images_train is None:
        total_images_train = np.sum(num_attributes_train)

    data_bog = bog_tilde / np.sum(bog_tilde, axis=1, keepdims=True)
    pred_bog = bog_gt_o / np.sum(bog_gt_o, axis=1, keepdims=True)

    p_t_a = np.zeros_like(data_bog)
    p_t_a = bog_tilde_train / np.expand_dims(num_attributes_train, 0)
    p_t = np.sum(bog_tilde_train, axis=1)/total_images_train

    origs_all = []

    diff = np.zeros_like(data_bog)
    for i in range(len(data_bog)):
        for j in range(len(data_bog[0])):
            diff[i][j] = pred_bog[i][j] - data_bog[i][j]
            indicator = np.sign(p_t_a[i][j] - p_t[i])
            if indicator == 0:
                diff[i][j] = 0
            elif indicator == -1:
                diff[i][j] = - diff[i][j]

    if disaggregate:
        diff_before = diff.copy()
    value = np.nanmean(diff)
    if toprint:
        print("Task->Attribute: {}".format(value))
    if disaggregate:
        return diff_before, value#, origs_all
    return value

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



