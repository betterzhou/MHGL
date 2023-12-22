import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd


def top_K_precision_recall(pred_top_k_gnd, K, total_anomaly_num):
    """
    1 --> anomaly
    0 --> normal
    """
    top_K_anomaly_num = sum(list(pred_top_k_gnd)[:K])
    top_K_pred_num = int(K)

    Precision_K = float(top_K_anomaly_num / top_K_pred_num)
    Recall_K = float(top_K_anomaly_num / total_anomaly_num)

    return Precision_K, Recall_K


def AD_metric(pred_anomaly_score, gnds):
    """
    pred_anomaly_score: numpy array;
    gnds: numpy array; 0->normal; 1->anomaly
    """
    roc_auc = roc_auc_score(gnds, pred_anomaly_score)
    roc_pr_area = average_precision_score(gnds, pred_anomaly_score)

    # precision@K
    sorted_index = np.argsort(-pred_anomaly_score, axis=0)
    top_ranking_gnd = gnds[sorted_index]

    # top K metric
    test_outliers_num = (np.where(gnds==1)[0]).shape[0]
    all_results = []
    pre_k = []
    rec_k = []
    for K in [100, 200, 300, 400, 500, 600]:
        precision_k, recall_k = top_K_precision_recall(top_ranking_gnd, K, test_outliers_num)
        pre_k.append(precision_k)
        rec_k.append(recall_k)
    pre_k.extend(rec_k)
    all_results.extend(pre_k)

    # save results
    all_results.append(roc_auc)
    all_results.append(roc_pr_area)

    print('\n\n ----------------------------',  'overall evaluation')
    print('Precision_100', 'Precision_200', 'Precision_300',
          'Precision_400', 'Precision_500', 'Precision_600',
          'Recall_100', 'Recall_200', 'Recall_300',
          'Recall_400', 'Recall_500', 'Recall_600',
          'AUC', 'AUPR')
    print(all_results)
    print('----------------------------',)
    return all_results


def get_intersection_index(target_list, big_list):
    """
    e.g.,
    target_list = [8,234,19,1,5,119,311]
    big_list = [i for i in range(20, 500)]

    local_indices_in_target_list [1, 6, 5]
    """
    ind_dict = dict((k, i) for i, k in enumerate(target_list))
    intersection = set(ind_dict) & set(big_list)
    local_indices_in_target_list = [ind_dict[x] for x in intersection]
    return local_indices_in_target_list


def metric_unseen_RCD(pred_test_anomaly_score, gnds_test, test_indices, all_seen_outlier_indices):
    total_outliers = np.where(gnds_test == 1)[0]
    seen_outlier_local_indices_in_testing = get_intersection_index(target_list=test_indices.tolist(),
                                                                   big_list=all_seen_outlier_indices.tolist())
    gnds_test[seen_outlier_local_indices_in_testing] = 0
    unseen_outliers = np.where(gnds_test==1)[0]
    print('unseen_outliers in test', unseen_outliers.shape[0])

    # precision@K
    sorted_index = np.argsort(-pred_test_anomaly_score, axis=0)
    top_ranking_gnd = gnds_test[sorted_index]

    # top K metric
    test_outliers_num = (np.where(gnds_test == 1)[0]).shape[0]
    print('\n\n ----------------------------', 'unseen RCD number in testing:', test_outliers_num)
    all_results = []
    pre_k = []
    rec_k = []
    for K in [100, 200, 300, 400, 500, 600]:
        precision_k, recall_k = top_K_precision_recall(top_ranking_gnd, K, test_outliers_num)
        pre_k.append(precision_k)
        rec_k.append(recall_k)
    pre_k.extend(rec_k)
    all_results.extend(pre_k)

    print('----------------------------', 'unseen evaluation')
    print('Precision_100', 'Precision_200', 'Precision_300',
          'Precision_400', 'Precision_500', 'Precision_600',
          'Recall_100', 'Recall_200', 'Recall_300',
          'Recall_400', 'Recall_500', 'Recall_600',
          )
    print(all_results)
    print('----------------------------',)
    return all_results


