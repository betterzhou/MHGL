import numpy as np
import random
from sklearn.mixture import GaussianMixture


def GMM_infer_abnormal_distribution(sampled_outlier_indx, attri_mat, outlier_cluster_num, return_type="list"):

    outlier_attri_mat = attri_mat[sampled_outlier_indx]
    gmm = GaussianMixture(n_components=outlier_cluster_num, covariance_type='full').fit(outlier_attri_mat)
    outlier_cluster_indx = gmm.predict(outlier_attri_mat)

    empty_cluster_Flag = False  # Flag: whether exist empty cluster

    support_set_dict = dict()
    support_set_list = []
    if return_type == "dict":
        for i in range(outlier_cluster_num):
            cluster_temp = np.where(outlier_cluster_indx == i)[0]
            if len(cluster_temp) == 0:
                empty_cluster_Flag = True
            support_set_dict['outlier_set_' + str(i)] = sampled_outlier_indx[cluster_temp]
        for (k, v) in support_set_dict.items():
            print('dict[%s] = ' % k, v)

    if return_type == "list":
        for i in range(outlier_cluster_num):
            cluster_temp = np.where(outlier_cluster_indx == i)[0]
            if len(cluster_temp) == 0:
                empty_cluster_Flag = True
            support_set_list.append(sampled_outlier_indx[cluster_temp])

    return support_set_list, empty_cluster_Flag


def find_center_from_cluster(anomaly_supp_sets, Emb_matrix):
    centers_list = []
    # find center in anomaly support set
    for cluster_k_node_indices in anomaly_supp_sets:
        emb_mat = Emb_matrix[cluster_k_node_indices]
        center = np.mean(emb_mat, axis=0)
        centers_list.append(center)
    return centers_list


def Euclidien_distance(vec1, vec2):
    Euclidien_score = np.linalg.norm(vec1 - vec2)

    return Euclidien_score


def GMM_infer_normal_distribution(normal_nodes, attri_mat, cluster_num):
    normal_attri_mat = attri_mat[normal_nodes]
    gmm = GaussianMixture(n_components=cluster_num, covariance_type='full').fit(normal_attri_mat)  # clustering
    normal_cluster_indx = gmm.predict(normal_attri_mat)

    support_set_list = []
    for i in range(cluster_num):
        cluster_temp = np.where(normal_cluster_indx == i)[0]
        support_set_list.append(list(normal_nodes[cluster_temp]))

    return support_set_list


def GMM_infer_normal_distribution_loop_splitting(selected_normal_nodes, attribute_mat, normal_clu_num,
                                                cluster_max_thrshold=150, cluster_min_thrshold=30):
    support_set_2 = GMM_infer_normal_distribution(selected_normal_nodes, attribute_mat, cluster_num=normal_clu_num)
    support_set_2.sort(key=lambda x: len(x), reverse=True)
    support_set_final = []
    support_set_tmp = []
    whether_keep_split = True
    support_set_init = support_set_2  # just a initial

    while whether_keep_split:
        for cluster_nodes_list in support_set_init:
            cluster_nodes_num = len(cluster_nodes_list)

            # if a cluster is larger than cluster_threshold, further split the cluster
            if cluster_nodes_num > cluster_max_thrshold:
                sub_cluster_num = cluster_nodes_num // cluster_max_thrshold + 1
                # clustering again
                sub_clusters = GMM_infer_normal_distribution(np.array(cluster_nodes_list), attribute_mat, cluster_num=sub_cluster_num)
                support_set_tmp.extend(sub_clusters)
            elif cluster_max_thrshold >= cluster_nodes_num and cluster_nodes_num >= cluster_min_thrshold:
                support_set_tmp.append(cluster_nodes_list)
            else:
                print('fewer than', cluster_min_thrshold, '\t delete this cluster...')

        support_set_tmp.sort(key=lambda y: len(y), reverse=True)

        # control whether stop loop
        qualified_clu_num_count = 0
        for clu_list in support_set_tmp:
            if len(clu_list) <= cluster_max_thrshold:
                qualified_clu_num_count += 1
        if qualified_clu_num_count == len(support_set_tmp):
            whether_keep_split = False
        else:
            support_set_init = support_set_tmp
            support_set_tmp = []  # !!! clear out

    for k in range(len(support_set_tmp)):
        if len(support_set_tmp[k]) >= cluster_min_thrshold:
            support_set_final.append(np.array(support_set_tmp[k]))
    return support_set_final


def select_most_K_percent_cloest_distance(center_node_emb, node_list, attri_mat, top_k_node):

    def Euclidien_distance(vec1, vec2):
        Euclidien_score = np.linalg.norm(vec1 - vec2)
        return Euclidien_score

    Euclidien_distance_list = []
    for j in node_list:
        similarity_score_j = Euclidien_distance(center_node_emb, attri_mat[j])
        Euclidien_distance_list.append(similarity_score_j)

    sorted_index = np.argsort(np.array(Euclidien_distance_list))
    return list(np.array(node_list)[sorted_index])[0:top_k_node], list(np.array(Euclidien_distance_list)[sorted_index])[top_k_node]


def select_cloest_node_smaller_than_threshold(center_node_emb, node_list, attri_mat, threshold):

    def Euclidien_distance(vec1, vec2):
        Euclidien_score = np.linalg.norm(vec1 - vec2)
        return Euclidien_score

    def find_node_indx_smaller_than_threshold(node_indx_array, node_distance_list, threshold_dis):
        required_indx_local = np.where(node_distance_list < threshold_dis)[0]
        final_node_indx = node_indx_array[required_indx_local]
        return final_node_indx

    Euclidien_distance_list = []
    for j in node_list:
        similarity_score_j = Euclidien_distance(center_node_emb, attri_mat[j])
        Euclidien_distance_list.append(similarity_score_j)

    final_nodes_indx_array = find_node_indx_smaller_than_threshold(node_list, np.array(Euclidien_distance_list), threshold)

    return final_nodes_indx_array


def find_center_nodes_from_cluster_based_on_distribution(normal_node_cluster, emb_matrix, distribution_prob=0.3):
    normal_support_set_final = []
    normal_hypersphere_centers = []

    total_nodes_num = emb_matrix.shape[0]
    total_nodes_indx = np.array([i for i in range(total_nodes_num)])

    for list_k in normal_node_cluster:
        # get cluster centroid
        mat_k = emb_matrix[np.array(list_k)]
        center_node_emb = np.mean(mat_k, axis=0)
        normal_hypersphere_centers.append(center_node_emb)

        cluster_total_node_num = len(list_k)
        threshold_node = int(distribution_prob * cluster_total_node_num)  # 0.7 * 100 = 70
        _, percent_70_cloest_dis = select_most_K_percent_cloest_distance(center_node_emb, list_k, emb_matrix, threshold_node)

        high_prob_unlabeld_node = select_cloest_node_smaller_than_threshold(center_node_emb, total_nodes_indx, emb_matrix, percent_70_cloest_dis)

        if high_prob_unlabeld_node.shape[0] >= 200:
            high_prob_unlabeld_node = high_prob_unlabeld_node[0:200]
        normal_support_set_final.append(high_prob_unlabeld_node)

    return normal_hypersphere_centers, normal_support_set_final
