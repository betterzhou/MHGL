# Unseen Anomaly Detection on Networks via Multi-Hypersphere Learning (MHGL, SDM 2022)

## 1.Introduction
This repository contains code for the paper "[Unseen Anomaly Detection on Networks via Multi-Hypersphere Learning](https://epubs.siam.org/doi/pdf/10.1137/1.9781611977172.30)" (SDM 2022).

## 2. Usage
### Requirements:
+ pytorch==1.7.0
+ scikit-learn
+ scipy
+ pandas
+ numpy

### Datasets:
Users can use the dataset provided in the folder. Users can also build their own datasets by following the setup in the paper.

The processed datasets are put into the ./Open_datasets/ folder.

### Data Format:
The input data for MHGL is a '.mat' file with 'gnd' (ground-truth), 'Attributes' (attributes), and 'Network' (graph structure).

It also requires csv files to indicate the nodes from the seen rare-categories (known anomaly types) and unseen rare-categories (unknown anomaly types). Because it simulates the scenarios where labeled anomalies only cover limited types and novel anomaly types exist in the test set. 

Please refer to the provided dataset in this repo. 

### Data Split:
Please note that sampling different labeled anomalies and labeled normal data for each run may cause large performance variances. 
It is suggested to fix the training labels.

### Example:
+ cd ./src/

+ python main.py --dataset=AD_amazon_electronics_computers --max_used_seen_anomalies=20  --outlier_clu_num=1 --ratio_labeled_normal=0.1 --total_clu_num=10 --nodes_in_cluster=100  --cluster_split_threshold=100  --whether_aug=True --data_aug_factor=1 --feature_extractor=GCN --labeld_seen_outliers_num=20 --lr=0.0001 --abnormal_weights=0.001 --normal_weights=1.0 --seed=1
+ python main.py --dataset=AD_amazon_electronics_photo --max_used_seen_anomalies=20  --outlier_clu_num=1 --ratio_labeled_normal=0.1 --total_clu_num=10 --nodes_in_cluster=100  --cluster_split_threshold=50 --cluster_min_threshold=30  --whether_aug=True --data_aug_factor=1 --feature_extractor=GCN --labeld_seen_outliers_num=20 --lr=0.0001 --abnormal_weights=1000 --normal_weights=1.0 --seed=1
+ python main.py --dataset=AD_ms_academic_cs --max_used_seen_anomalies=20  --outlier_clu_num=1 --ratio_labeled_normal=0.1 --total_clu_num=20 --nodes_in_cluster=100  --cluster_split_threshold=100  --whether_aug=True --data_aug_factor=1 --feature_extractor=GCN --labeld_seen_outliers_num=20 --lr=0.001 --abnormal_weights=10 --normal_weights=1.0 --seed=1


## 3. Citation
Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@inproceedings{zhou2022unseen,
  title={Unseen anomaly detection on networks via multi-hypersphere learning},
  author={Zhou, Shuang and Huang, Xiao and Liu, Ninghao and Tan, Qiaoyu and Chung, Fu-Lai},
  booktitle={Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)},
  pages={262--270},
  year={2022},
  organization={SIAM}
}
```

