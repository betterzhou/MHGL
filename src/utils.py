import numpy as np
import scipy.sparse as sp
import torch
import scipy.io
from sklearn.model_selection import train_test_split
import random
import copy


def overall_normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def load_data_AD(root_path, data_source, feat_norm=False):
    data = scipy.io.loadmat(root_path + data_source + '.mat')
    adj_csr_matrix = sp.csr_matrix(data["Network"], dtype=np.float32)
    attributes = sp.csr_matrix(data["Attributes"], dtype=np.float32)
    attri_matrix = attributes.todense()
    if feat_norm:
        attri_matrix = overall_normalization(np.array(attri_matrix))
    label_binary = data["gnd"]
    print('data_source:', data_source)
    return adj_csr_matrix, attri_matrix, label_binary


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def modify_label_func(label):
    new_y = []
    for j in label:
        if j == 0:
            new_y.append(0)
        if j == 1:
            new_y.append(1)
    return np.array(new_y)


def shuffle_split(return_node_num, total_node, seed):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)
    return randomlist[:return_node_num], randomlist[return_node_num: total_node]


def format_to_sparse_torch_for_GCN(adj_csr_matrix, features, labels):
    """
    features: dense matrix
    labels: one-hot format
    adj: csr_matrix
    """
    features = torch.FloatTensor(features)
    labels = torch.from_numpy(labels)

    adj_normed_sp = normalize(adj_csr_matrix + sp.eye(adj_csr_matrix.shape[0]))  # based on GCN paper to norm adj matrix
    adj = sparse_mx_to_torch_sparse_tensor(adj_normed_sp)

    return adj, features, labels


def euclidean_distance(node_embedding, c):
    return torch.sum((node_embedding - c) ** 2)


def nor_loss(node_embedding_list, c):
    s = 0
    num_node = node_embedding_list.size()[0]
    for i in range(num_node):
        s = s + euclidean_distance(node_embedding_list[i], c)
    return s


def euclidean_distance_loss_v2_MixUp(all_center_torch_list, support_sets, hidden_emb_matrix,
                                          data_aug_factor, MIXUP_lamda, whether_aug=False):
    loss_tmp = 0
    num_node = 0
    loss_mixup = 0
    num_node_aug = 0

    for j in range(len(all_center_torch_list)):
        center_vec_j = all_center_torch_list[j]
        cluster_j_node_indices = support_sets[j]
        loss_tmp = loss_tmp + nor_loss(hidden_emb_matrix[cluster_j_node_indices], center_vec_j)
        num_node += support_sets[j].shape[0]

    Nloss = loss_tmp
    final_loss = Nloss/num_node

    if whether_aug == True:
        generated_emb_for_all_clusters = data_aug_sampling(hidden_emb_matrix, support_sets, data_aug_factor, MIXUP_lamda)
        assert len(generated_emb_for_all_clusters) == len(generated_emb_for_all_clusters)

        for j in range(len(generated_emb_for_all_clusters)):  # for each normal center
            center_vec_j = all_center_torch_list[j]
            for k in range(len(generated_emb_for_all_clusters[j])):
                cluster_j_virtual_emb_k = generated_emb_for_all_clusters[j][k]
                loss_mixup = loss_mixup + euclidean_distance(cluster_j_virtual_emb_k, center_vec_j)
                num_node_aug += 1

        final_loss = loss_mixup/num_node_aug

    return final_loss


def euclidean_distance_loss_outliers_to_normal_center_v2_MixUp(all_center_torch_list, support_sets, hidden_emb_matrix,
                                          data_aug_factor, MIXUP_lamda, whether_aug=False):
    loss_tmp = 0
    assert len(support_sets) == 1  # assume only 1 abnormal support set
    num_node = support_sets[0].shape[0]
    loss_mixup = 0
    num_node_aug = 0

    for i in range(len(all_center_torch_list)):
        center_vec_j = all_center_torch_list[i]
        cluster_j_node_indices = support_sets[0]
        loss_tmp = loss_tmp + nor_loss(hidden_emb_matrix[cluster_j_node_indices], center_vec_j)

    Nloss = loss_tmp
    final_loss = Nloss/(num_node * len(all_center_torch_list))

    if whether_aug == True:
        generated_emb_for_all_clusters = data_aug_sampling(hidden_emb_matrix, support_sets, data_aug_factor, MIXUP_lamda)

        for j in range(len(all_center_torch_list)):  # for each normal center
            center_vec_j = all_center_torch_list[j]
            for k in range(len(generated_emb_for_all_clusters[0])):  # only 1 outlier cluster
                cluster_j_virtual_emb_k = generated_emb_for_all_clusters[0][k]
                loss_mixup = loss_mixup + euclidean_distance(cluster_j_virtual_emb_k, center_vec_j)
                num_node_aug += 1

        final_loss = Nloss/(num_node * len(all_center_torch_list)) + loss_mixup/num_node_aug

    return final_loss


def data_aug_sampling(hidden_emb, cluster_nodes_indx_list, data_aug_factor, lamda):
    nodes_num_for_average = 2
    generated_emb_for_all_clusters = []  # a list of list
    for i in range(len(cluster_nodes_indx_list)):   # for each cluster
        nodes_num_in_this_cluster = cluster_nodes_indx_list[i].shape[0]  # e.g., 30

        generated_emb_for_1_cluster = []  # a list of emb vectors
        for j in range(nodes_num_in_this_cluster):
            # repeat sampling
            for k in range(data_aug_factor):
                sampled_local_indx = sample_random_node_index(nodes_num_for_average,
                                                          total_node=nodes_num_in_this_cluster,
                                                          seed=100*i+20*j+k)
                sampled_2_nodes_global_indx = np.array(cluster_nodes_indx_list[i])[sampled_local_indx]

                generated_emb_jk = data_aug_linear_mix_up(hidden_emb, sampled_2_nodes_global_indx, lamda)
                generated_emb_for_1_cluster.append(generated_emb_jk)

        generated_emb_for_all_clusters.append(generated_emb_for_1_cluster)

    return generated_emb_for_all_clusters


def sample_random_node_index(return_node_num, total_node, seed):
    randomlist = [i for i in range(total_node)]
    random.seed(seed)
    random.shuffle(randomlist)
    return randomlist[:return_node_num]


def data_aug_linear_mix_up(all_emb, base_nodes_indices, lamda):
    vec1 = all_emb[base_nodes_indices][0]
    vec2 = all_emb[base_nodes_indices][1]
    permuted_vec = lamda * vec1 + (1-lamda) * vec2
    return permuted_vec


def calculate_distan_to_closest_normal_center_v2(normal_center_torch_list, all_test_index, hidden_emb):
    test_node_num = all_test_index.shape[0]
    normal_cluster_num = len(normal_center_torch_list)
    distance_matrix = np.zeros((test_node_num, normal_cluster_num))

    for k in range(len(all_test_index)):
        node_k_global_indx = all_test_index[k]
        node_k_position_vec = hidden_emb[node_k_global_indx]
        for j in range(len(normal_center_torch_list)):
            center_vec_j = normal_center_torch_list[j]
            distance_matrix[k, j] = euclidean_distance(node_k_position_vec, center_vec_j).item()

    dis_to_cloest_center = []
    for i in range(len(all_test_index)):
        dis_to_cloest_center.append(min(distance_matrix[i]))

    return dis_to_cloest_center


def RCD_data_split_szhou_v2_fix_test_anomaly_for_exp_analys(y, known_abnoral_class_indx, unknown_abnoral_class_indx,
                         random_state, labeld_outliers_num, ratio_labeled_normal, exclude_anomalies_num, exclude_normal_ratio):
    idx_norm = np.where(y == 0)[0]
    idx_out = np.where(y == 1)[0]
    idx_out_known = known_abnoral_class_indx
    idx_out_unknown = unknown_abnoral_class_indx
    known_RCD_total_num = idx_out_known.shape[0]

    # some normal nodes are labeled normal data in train,
    train_norm_indx, test_norm_indx = train_test_split(idx_norm, test_size=1 - ratio_labeled_normal, random_state=random_state)
    # a few labeled outliers are from the known rare-category
    train_outliers_num = labeld_outliers_num  # default 20

    shuffled_outlier_train, shuffle_outlier_test = shuffle_split(train_outliers_num, known_RCD_total_num, random_state)
    known_outlier_train_indx = idx_out_known[np.array(shuffled_outlier_train)]  # !!! get trn_index from the known-class

    known_outlier_test_indx = idx_out_known[np.array(shuffle_outlier_test)]  # !!! get test_index from the known-class
    total_outlier_test_indx = np.append(known_outlier_test_indx, idx_out_unknown)  # !!! combine test_known and all_unknown

    train_out_indx = known_outlier_train_indx
    test_out_indx = total_outlier_test_indx
    # _______________________________________________________

    # ____________________________________________________________________
    total_testing_anomaly_num = test_out_indx.shape[0]
    remained_testing_anomaly_num = total_testing_anomaly_num - exclude_anomalies_num
    # to delete random sampled anomalies
    remained_outlier_local_indx = sample_random_node_index(remained_testing_anomaly_num, total_testing_anomaly_num, random_state)
    fixed_anomaly_list = test_out_indx[remained_outlier_local_indx]
    final_test_out = np.array(fixed_anomaly_list)
    # ____________________________________________________________________

    exclude_normal_node_num = int(exclude_normal_ratio * idx_norm.shape[0])  # here, ratio * normal_total_num
    remained_test_normal_num = test_norm_indx.shape[0] - exclude_normal_node_num
    if exclude_normal_node_num == 0:
        final_test_normal = test_norm_indx
    else:
        remained_normal_local_indx = sample_random_node_index(remained_test_normal_num, test_norm_indx.shape[0], random_state)
        fixed_test_normal_list = test_norm_indx[remained_normal_local_indx]
        final_test_normal = np.array(fixed_test_normal_list)

    # merge test data
    all_test_indx = np.concatenate((final_test_normal, final_test_out))  # rest normal + rest outliers

    return train_out_indx, train_norm_indx, all_test_indx, final_test_out, final_test_normal









