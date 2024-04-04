# Exploring CICIDS2017 Dataset using EDA and Machine Learning Models

## Data Source
The dataset used in this analysis was obtained from the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html). It comprises CSV files containing network traffic data.

## Exploratory Data Analysis (EDA)
Initial data exploration was performed using the `CICIDS2017_EDA.ipynb` file.

## Feature Significance for Attack Classification
To identify significant features for classifying attacks versus non-attacks (BENIGN), the data was filtered to include the two days with the highest attack frequency. A Decision Tree (DT) model was trained and achieved a mean precision and recall of 0.99 on the test data (using 5-fold cv). The top 10 important features from the DT model are: 'Average Packet Size', 'Avg Bwd Segment Size', 'Bwd Header Length','Bwd IAT Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Mean','Flow IAT Min', 'Flow IAT Std', 'Fwd IAT Mean', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward', 'Max Packet Length', 'Packet Length Mean', 'Subflow Bwd Bytes','Total Fwd Packets', 'Total Length of Bwd Packets'. 

The implementation code is available in the `Binary_attack_classification.py` file, and the confusion matrix figure is provided in `dt_confusion_matrix.png`.

## Cluster Analysis of Attack Types
K-means clustering (k=4) was utilized to group various attack types into four clusters. The identified clusters are as follows:
- **Cluster 0**: PortScan, Infiltration, Web Attack Brute Force, Web Attack XSS, Web Attack Sql Injection, SSH-Patator. 
- **Cluster 1**: DoS Slowhttptest, DoS Hulk, DoS GoldenEye.
- **Cluster 2**: DDoS, Bot. 
- **Cluster 3**: FTP-Patator, DoS slowloris, Heartbleed.
- 
The implementation code can be found in the `Label_kmeans.py` file.

## Classification of Attack Types Using XGBoost
Attacks were classified into the four clusters using XGBoost, achieving an average precision and recall of 0.99 on the test data (using 5-fold cv). The top 10 significant features from the XGBoost model are: 'Average Packet Size', 'Bwd Header Length', 'Flow Bytes/s', 'Fwd Header Length', 'Fwd Packet Length Max', 'Init_Win_bytes_backward', 'Total Fwd Packets', 'Total Length of Bwd Packets', 'Total Length of Fwd Packets', 'act_data_pkt_fwd', 'min_seg_size_forward'. 

The duration flow didn't emerge as a significant feature, contrary to my initial expectation based on the EDA.

The implementation code is available in the `Classification_of_Attack_Types.py` file, and the confusion matrix figure is provided in `xgboost_confusion_matrix.png`.


