Applying Machine Learning Models and EDA on CICIDS2017 Dataset

The csv files were downloaded from  https://www.unb.ca/cic/datasets/ids-2017.html

At first I examined at the data using EDA. The EDA appears inCICIDS2017 EDA.ipynb file.
Next I decided to explore which features in the data are significant for classifying between attacks to non-attacks (BENIGN). Therefore I filtered the data to deal with the 2 days that have most of the attacks so my data will be almost balanced. Then I ran DT and got a mean precision and mean recall of 0.99 whuch mean that I can classify very good between attack and non attack. 
The union of the top 10 features which as the highest importance of the DT model are:
'Average Packet Size', 'Avg Bwd Segment Size', 'Bwd Header Length','Bwd IAT Mean', 'Bwd Packet Length Max', 'Bwd Packet Length Mean','Flow IAT Min', 'Flow IAT Std', 'Fwd IAT Mean', 'Init_Win_bytes_backward', 'Init_Win_bytes_forward', 'Max Packet Length', 'Packet Length Mean', 'Subflow Bwd Bytes','Total Fwd Packets', 'Total Length of Bwd Packets'.

The code appears in Binary attack classification.py file and the confusion matrix figure is attached in a separate file named dt_confusion_matrix.png.

Second, because there are many types of attacks I tried to merge them to 4 clusters using k-means (k=4). 
I got that the attacks are divided to the following 4 clusters:
cluster 0 - PortScan, Infiltration, Web Attack Brute Force, Web Attack XSS, Web Attack Sql Injection, SSH-Patator
cluster 1 - DoS Slowhttptest, DoS Hulk, DoS GoldenEye
cluster 2 - DDoS, Bot
cluster 3 - FTP-Patator, DoS slowloris, Heartbleed
The code appear in Label kmeans.py file.

Last, we will use the new clusters to classify between these 4 clusters of atteck using our data
