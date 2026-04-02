import os
import numpy as np
import torch
import random
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    confusion_matrix, 
    accuracy_score
)


def get_Pearson_adj(sub_region_series,threshold,numnodes=90):
    subj_fc_adj = np.corrcoef(np.transpose(sub_region_series))
    subj_fc_adj_up=subj_fc_adj[np.triu_indices(numnodes,k=1)]
    subj_fc_adj_list = subj_fc_adj_up.reshape((-1))
    thindex = int(threshold * subj_fc_adj_list.shape[0])
    thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex-1]]
    subj_fc_adj_t = np.zeros((numnodes, numnodes))
    subj_fc_adj_t[subj_fc_adj > thremax] = 1
    subj_fc_adj=subj_fc_adj_t
    return subj_fc_adj

def save_args(file,args):
    with open(file, "a+", encoding='utf-8') as rf:
        argsDict = args.__dict__
        for eachArg, value in argsDict.items():
            rf.writelines(eachArg + ' : ' + str(value) + ' ')
        rf.write('\n')


def save_each_fold(file,test_acc,test_sen,test_spe,test_f1,test_auc,test_aucprop=None):
    with open(file, "a+", encoding='utf-8') as rf:
        if test_aucprop==None:
            rf.write('#acc:%.6f sensitivity:%.6f specificity:%.6f f1:%.6f auc:%.6f\n' % (test_acc,test_sen,test_spe,test_f1,test_auc))
        else:
            rf.write('#acc:%.6f sensitivity:%.6f specificity:%.6f f1:%.6f auc:%.6f auc_prop:%.6f\n' % (test_acc,test_sen,test_spe,test_f1,test_auc,test_aucprop))


def save_std(file, 
             acc_iter, sen_iter, spe_iter, f1_iter, auc_iter, aucprop_iter=None):
    with open(file, "a+", encoding='utf-8') as rf:
        if aucprop_iter==None:
            rf.write('[average acc:%.2f (± %.2f) sensitivity:%.2f (± %.2f) specificity:%.2f (± %.2f) f1:%.2f (± %.2f) auc:%.2f (± %.2f)]\n' %
               (np.mean(acc_iter), np.std(acc_iter), 
                np.mean(sen_iter), np.std(sen_iter),
                np.mean(spe_iter), np.std(spe_iter),
                np.mean(f1_iter),  np.std(f1_iter),
                np.mean(auc_iter), np.std(auc_iter)))
        else:
            rf.write('[average acc:%.2f (± %.2f) sensitivity:%.2f (± %.2f) specificity:%.2f (± %.2f) f1:%.2f (± %.2f) auc:%.2f (± %.2f) auc_prop:%.2f (± %.2f)]\n' %
               (np.mean(acc_iter),      np.std(acc_iter), 
                np.mean(sen_iter),      np.std(sen_iter),
                np.mean(spe_iter),      np.std(spe_iter),
                np.mean(f1_iter),       np.std(f1_iter),
                np.mean(auc_iter),      np.std(auc_iter),
                np.mean(aucprop_iter),  np.std(aucprop_iter)))


def save_pred(file, y, pred, predpro=None):
    with open(file, "a+", encoding='utf-8') as rf:
        rf.write('y:%s\n' % str(y))
        rf.write('pred:%s\n' % str(pred))
        # if predpro:
        #     rf.write('predpro:%s\n' % str(predpro))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def evaluate_metrics(Y_test, Y_pred):
    con_mat = confusion_matrix(Y_test, Y_pred)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    if tn == 0 and fp == 0:
        specificity = 0
    else:
        specificity = tn / (fp + tn)
    if tp == 0 and fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    f1 = f1_score(Y_test, Y_pred)
    auc = roc_auc_score(Y_test, Y_pred)
    return sensitivity, specificity, f1, auc


# for baseline
def compute_metrics(test_y, pre):
    accuracy = accuracy_score(test_y, pre)
    f1 = f1_score(test_y, pre)
    auc = roc_auc_score(test_y, pre)
    con_mat = confusion_matrix(test_y, pre)
    tp = con_mat[1][1]
    fp = con_mat[0][1]
    fn = con_mat[1][0]
    tn = con_mat[0][0]
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    return accuracy, sensitivity, specificity, f1, auc


def max_min_norm(sub_region_series):
    subj_fc_mat_list = sub_region_series.reshape((-1))
    subj_fc_feature = (sub_region_series - min(subj_fc_mat_list)) / (max(subj_fc_mat_list) - min(subj_fc_mat_list))
    return subj_fc_feature


def get_Degree_fc(subj_fc_adj):
    rowsum = np.array(subj_fc_adj.sum(1))
    N = np.diag(rowsum)
    return N


def get_adj(connectivity,threshold):
    num_nodes = connectivity.shape[0]
    connectivity[np.isinf(connectivity)] = 1
    connectivity[np.isnan(connectivity)] = 0
    subj_fc_adj_up = connectivity[np.triu_indices(num_nodes, k=1)]  # 对角线及其以下都是0
    subj_fc_adj_list = subj_fc_adj_up.reshape((-1))
    thindex = int(threshold * subj_fc_adj_list.shape[0])
    thremax = subj_fc_adj_list[subj_fc_adj_list.argsort()[-1 * thindex - 1]]
    adj = np.zeros((num_nodes, num_nodes))
    adj[connectivity > thremax] = 1
    return adj

