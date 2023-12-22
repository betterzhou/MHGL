from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from models_GCN import *
from evaluation_AD import AD_metric, metric_unseen_RCD
import pandas as pd
from build_support_set import *


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='Disables CUDA training.')
parser.add_argument("--cuda_id", type=str, default="3", help='assign CUDA ID.')
parser.add_argument("--root_path", type=str, default="../Open_datasets/")
parser.add_argument('--dataset', help = 'dataset name', type = str, default='AD_ms_academic_cs')
parser.add_argument('--seed', help = 'random seed', type = int, default =23)
parser.add_argument("--epochs", type=int, default=300, help="training epochs")
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
parser.add_argument("--feature_extractor", type=str, default="GCN", help='feature extractor')
parser.add_argument("--outlier_clu_num", type=int, default=1, help="outlier hypersphere num")
parser.add_argument("--total_clu_num", type=int, default=10, help="total_clu_num")
parser.add_argument("--nodes_in_cluster", type=int, default=100, help="nodes_in_cluster")
parser.add_argument("--cluster_split_threshold", type=int, default=100,
                    help="if larger than it, further split the cluster")
parser.add_argument("--cluster_min_threshold", type=int, default=30, help="if fewer than it, delete the cluster")
parser.add_argument("--labeld_seen_outliers_num", type=int, default=20)
parser.add_argument("--ratio_labeled_normal", type=float, default=0.1)
parser.add_argument("--max_used_seen_anomalies", type=int, default=40, help="for sample efficiency discussion")
parser.add_argument("--max_used_labeled_normal", type=float, default=0.1, help="for sample efficiency discussion")
parser.add_argument("--whether_aug", type=bool, default=True, help="whether use MIXUP")
parser.add_argument("--data_aug_factor", type=int, default=1)
parser.add_argument("--lamda", type=float, default=0.5, help="weight factor in MIXUP")
parser.add_argument("--normal_weights", type=float, default=1, help="fix it as 1")
parser.add_argument("--abnormal_weights", type=float, default=1, help="loss_train")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

data_sets = args.dataset
random_seed = 1  # fixed
lr = args.lr
weight_decay = args.weight_decay
root_path = args.root_path


outlier_clu_num = args.outlier_clu_num
total_clu_num = args.total_clu_num
nodes_in_cluster = args.nodes_in_cluster
cluster_max_threshold = args.cluster_split_threshold
cluster_min_threshold = args.cluster_min_threshold

labeld_seen_outliers_num = args.labeld_seen_outliers_num
ratio_labeled_normal = args.ratio_labeled_normal
train_epoches = args.epochs
feature_extractor = args.feature_extractor

normal_clu_num = total_clu_num - outlier_clu_num
whether_aug = args.whether_aug
data_aug_factor = args.data_aug_factor
lamda = args.lamda

normal_weights = args.normal_weights
abnormal_weights = args.abnormal_weights

max_used_seen_anomalies = args.max_used_seen_anomalies
max_used_labeled_normal = args.max_used_labeled_normal

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# ____________________________________________________________________
adj_csr_matrix, attri_matrix, label_binary = load_data_AD(root_path, data_sets)
gnds = modify_label_func(label_binary)
all_outlier_global_indices = np.where(gnds == 1)[0]
adj, features, gnds_torch = format_to_sparse_torch_for_GCN(adj_csr_matrix, attri_matrix, gnds)

# ____________________________________________________________________
known_rare_categeory_file = root_path + data_sets + '_known_class_indx.csv'
unknown_rare_categeory_file = root_path + data_sets + '_unknown_class_indx.csv'
idx_known_RCD_class = np.loadtxt(known_rare_categeory_file, delimiter='\n', dtype=int)
idx_unknown_RCD_class = np.loadtxt(unknown_rare_categeory_file, delimiter='\n', dtype=int)

# get labeled norm, labeled seen outliers
exclude_anomalies_num = max_used_seen_anomalies - labeld_seen_outliers_num
exclude_normal_ratio = max_used_labeled_normal - ratio_labeled_normal
train_out_indx, train_norm_indx, all_test_indx, test_out_indx, test_norm_indx = RCD_data_split_szhou_v2_fix_test_anomaly_for_exp_analys(gnds, idx_known_RCD_class, idx_unknown_RCD_class,
                                                    random_seed, labeld_seen_outliers_num, ratio_labeled_normal, exclude_anomalies_num, exclude_normal_ratio)
# ____________________________________________________________________

model = GCN_vanilla_4_layers(nfeat=attri_matrix.shape[1],
            nhid=args.hidden,
            dropout=args.dropout)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda('cuda:'+args.cuda_id)
    features = features.cuda('cuda:'+args.cuda_id)
    adj = adj.cuda('cuda:'+args.cuda_id)

# initial forward
model.eval()
emb_matrix = model(features, adj)
emb_matrix = emb_matrix.detach().cpu().numpy()
anomaly_support_sets, whether_null_cluster = GMM_infer_abnormal_distribution(train_out_indx,
                                                                             emb_matrix,
                                                                             outlier_cluster_num=outlier_clu_num,
                                                                             return_type="list")
outlier_cluster_center_list = find_center_from_cluster(anomaly_support_sets, emb_matrix)

normal_support_sets_rough = GMM_infer_normal_distribution_loop_splitting(train_norm_indx, emb_matrix, normal_clu_num,
                                cluster_max_thrshold=cluster_max_threshold, cluster_min_thrshold=cluster_min_threshold)

normal_cluster_center_list, normal_support_set_final = find_center_nodes_from_cluster_based_on_distribution(normal_support_sets_rough, emb_matrix)

# ____________________________________________________________________
device = features.device

all_normal_center_torch_list = []
all_outlier_center_torch_list = []
for i in range(len(normal_cluster_center_list)):
    normal_center_i = torch.FloatTensor(normal_cluster_center_list[i]).to(device)
    all_normal_center_torch_list.append(normal_center_i)

for k in range(len(outlier_cluster_center_list)):
    outlier_center_k = torch.FloatTensor(outlier_cluster_center_list[k]).to(device)
    all_outlier_center_torch_list.append(outlier_center_k)
# ____________________________________________________________________


def train_model(normal_center_torch_list, normal_support_sets,
                outlier_center_torch_list, anomaly_support_sets):
    model.train()
    optimizer.zero_grad()
    hidden_emb = model(features, adj)
    normal_cluster_loss_all = euclidean_distance_loss_v2_MixUp(normal_center_torch_list, normal_support_sets, hidden_emb,
                                          data_aug_factor, lamda, whether_aug=whether_aug)

    outlier_distance_to_normal_centers_loss = euclidean_distance_loss_outliers_to_normal_center_v2_MixUp(normal_center_torch_list, anomaly_support_sets, hidden_emb,
                                                data_aug_factor, lamda, whether_aug=whether_aug)
    # -----------------------------------------------------------------------------------------
    seen_outlier_loss = 1 / (abnormal_weights * outlier_distance_to_normal_centers_loss)
    loss_train = normal_weights * normal_cluster_loss_all + seen_outlier_loss
    loss_train.backward()
    optimizer.step()
    return loss_train.item(), seen_outlier_loss.item()


for epoch_i in range(train_epoches):
    trn_loss, trn_outlier_loss = train_model(all_normal_center_torch_list, normal_support_set_final,
                           all_outlier_center_torch_list, anomaly_support_sets)
    print('epochs:', epoch_i, '\t\t trn_loss:', trn_loss)
# ______________________________________________________________
model.eval()
test_emb = model(features, adj)
dist_to_closest_normal_center = calculate_distan_to_closest_normal_center_v2(all_normal_center_torch_list, all_test_indx, test_emb)
anomalyness = np.array(dist_to_closest_normal_center)
query_node_gnd = gnds[all_test_indx]
results_metric_both_seen_unseen_RCD = AD_metric(anomalyness, query_node_gnd)
results_metric_only_unseen_RCD = metric_unseen_RCD(anomalyness, query_node_gnd,
                                                         test_indices=all_test_indx,
                                                         all_seen_outlier_indices=idx_known_RCD_class)

df = pd.DataFrame(np.array(results_metric_both_seen_unseen_RCD).reshape(-1, 14),
                  columns=['Precision_100', 'Precision_200', 'Precision_300',
                            'Precision_400', 'Precision_500', 'Precision_600',
                            'Recall_100', 'Recall_200', 'Recall_300',
                            'Recall_400', 'Recall_500', 'Recall_600',
                            'AUC', 'AUPR'
                           ])
df.to_csv('../results_overall/' + 'results_' + data_sets + '_lr_' + str(lr)
          + '_NormalWeight_' + str(normal_weights)+'_AbnormalWeight_' + str(
    abnormal_weights) + '_UsedOutlier_' + str(labeld_seen_outliers_num) + '_seed_' + str(args.seed) + '.csv')


df = pd.DataFrame(np.array(results_metric_only_unseen_RCD).reshape(-1, 12),
                  columns=['Precision_100', 'Precision_200', 'Precision_300',
                            'Precision_400', 'Precision_500', 'Precision_600',
                            'Recall_100', 'Recall_200', 'Recall_300',
                            'Recall_400', 'Recall_500', 'Recall_600',
                           ])
df.to_csv('../results_only_unseen/' + 'results_' + data_sets + '_lr_' + str(lr)
          + '_NormalWeight_' + str(normal_weights)+'_AbnormalWeight_' + str(
    abnormal_weights) + '_UsedOutlier_' + str(labeld_seen_outliers_num) + '_seed_' + str(args.seed) + '.csv')
# ____________________________________________________________________
