# Summary of the sentence simplifcation
Files in the directory:

- Main: 

  * [embed_cluster.py](embed_cluster.py): Code to create SBERT vectors on the simplified sentences and run MiniBatch KMeans for clustering.
  * [evaluate_cluster.py](evaluate_cluster.py): Code to evaluate the quality of the KMeans clustering.
  * [baseline.py](baseline.py): Code to create SBERT vectors on the non-simplified sentences and run MiniBatch KMeans for clustering.
  


## Sentence simplifcation:

* To start, we use use two methods for sentence simplifcation, [`ABCD`](https://aclanthology.org/2021.acl-long.303.pdf) and [`DisSim`](https://aclanthology.org/W19-8662/)
* We use the Manifesto corpus, which has labels for every complex sentence. Each simplified sentence is assigned the orginal label.
* We use SBERT embeddings on each type of simplified sentences.
* We run KMeans and HDBScan clustering on the embeddings, for the HDBScan, we cluster on PCA+UMAP reductions run on the embeddings.
* To evaluate, we cluster on cluster sizes ```16,32,64,128,256,512,1024``` and for each cluster size, we assign the most occuring true label to all the clustered sentences, and then we calculate the accuracy vs the actual labels.

## Some analysis
* The HDBScan clustering doesn't work well, most sentences are treated as outliers and given the label ```-1```.
* We have a baseline on the KMeans, where we embed the non-simplified sentences, and this has slighly higher accuracy.

  | Cluster size        | Acc. on simplified (ABCD)      | Acc. on baseline          |
  | -----------         | -----------                    |-----------                |
  | 16                  |0.2539916063989593              |0.2855039520586868         |
  | 32                  |0.26902885106778374             |0.3250116237020199         |
  | 64                  |0.2856689328357053              |0.335989564498631          |
  | 128                 |0.30573924085918264             |0.3554915534431988         |
  | 256                 |0.321566289315968               |0.3782740094022834         |
  | 512                 |0.3381599120375389              |0.3987188097329132         |
  | 1024                |0.3545522122249237              |0.4158831430490262         |

* Increasing cluster size improves accuracy.
* To evaluate further, next task is to cluster on cluster IDs again, similar to bag of words and see if there are any improvements.



