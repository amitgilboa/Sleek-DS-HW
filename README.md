# Exploring CICIDS2017 Dataset using EDA and Machine Learning Models

## Data Source
The dataset used in this analysis was obtained from the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html). It comprises CSV files containing network traffic data.

## Exploratory Data Analysis (EDA)
Initial data exploration was performed using the `CICIDS2017_EDA.ipynb` file.

## Feature Significance for Attack Classification (insight 1)
To identify significant features for classifying attacks versus non-attacks (BENIGN), the data was filtered to include the two days with the highest attack frequency (Friday afternoon and Wednesday). A Decision Tree (DT) model of binary classifier was trained and achieved remarkably high results, with a mean precision and recall of 0.99 on the test data, as determined through 5-fold cross-validation. The top 10 important features from the DT model are: 'Average Packet Size', 'Avg Bwd Segment Size', 'Bwd Header Length','Bwd IAT Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Mean','Flow IAT Min', 'Flow IAT Std', 'Fwd IAT Mean', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward', 'Max Packet Length', 'Packet Length Mean', 'Subflow Bwd Bytes','Total Fwd Packets', 'Total Length of Bwd Packets'. 

The implementation code is available in the `Binary_attack_classification.py` file, and the confusion matrix figure is provided in `dt_confusion_matrix.png`.

## Cluster Analysis of Attack Types (insight 2)
K-means clustering with k=4 was employed to group various attack types into four clusters. This approach was chosen due to the presence of numerous attack types, some of which had only a few observations. The identified clusters are as follows:
- **Cluster 0**: PortScan, Infiltration, Web Attack Brute Force, Web Attack XSS, Web Attack Sql Injection, SSH-Patator. 
- **Cluster 1**: DoS Slowhttptest, DoS Hulk, DoS GoldenEye.
- **Cluster 2**: DDoS, Bot. 
- **Cluster 3**: FTP-Patator, DoS slowloris, Heartbleed.

The implementation code can be found in the `Label_kmeans.py` file.

## Classification of Attack Types Using XGBoost (insight 3)
Attacks were classified into the four clusters using XGBoost, achieving an average precision and recall of 0.99 on the test data (using 5-fold cv). The top 10 significant features from the XGBoost model are: 'Average Packet Size', 'Bwd Header Length', 'Flow Bytes/s', 'Fwd Header Length', 'Fwd Packet Length Max', 'Init_Win_bytes_backward', 'Total Fwd Packets', 'Total Length of Bwd Packets', 'Total Length of Fwd Packets', 'act_data_pkt_fwd', 'min_seg_size_forward'. 

The duration flow didn't emerge as a significant feature, contrary to my initial expectation based on the EDA.

The implementation code is available in the `Classification_of_Attack_Types.py` file, and the confusion matrix figure is provided in `xgboost_confusion_matrix.png`.

## Conclusion
In conclusion, this analysis delved into the CICIDS2017 dataset using EDA and machine learning models. I discovered significant insights into the dataset's features and the classification of attack types.

The initial EDA revealed essential insights into the dataset's structure and distributions. Subsequently, I identified significant features for classifying attacks versus non-attacks using a DT model. Additionally, I applied k-means clustering to group various attack types into four clusters, facilitating a more manageable classification approach. Utilizing XGBoost, I classified attacks into these clusters, achieving high precision and recall scores.

However, there is much more to explore in this dataset. Future investigations could include exploring the distribution of significant features obtained from the classification models. Furthermore, examining the merged data with the TrafficLabelling_ directory could provide additional insights into network traffic patterns and attack behaviors.

The code and visualizations generated during this analysis are available in the corresponding files. Moving forward, further exploration of the dataset promises to uncover more valuable insights into network security and attack detection.

___________________________________________________________________________________________________________________________________________

* The final dataset, merged_df2, used for running the models has been saved as a pickle file. You can download it from the following link: https://drive.google.com/file/d/1RQiO5IXi4yeZ_mScixlBq4rC8xk2oyqj/view?usp=sharing 


