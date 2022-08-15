# Summary of the paraphrase and sentence clustering
Files in the directory:

- Main: 

  * [data_prepreocessing.py](data_preprocessing.py): Code to create SBERT vectors for MRPC+NLI parapharase training data.
  * [dataloader_testcorpus.py](dataloader_testcorpus.py): Code to create SBERT vectors on test datsets like BBC, Trump, MemSum etc.
  * [logistic_classifier.py](logistic_classifier.py): Code to train a logistic regression paraphrase detection model.
  * [logistic_test.py](logistic_test.py): Code to test a logistic regression paraphrase detection model.
  * [logistic_test_sts.py](logistic_test_sts.py): Code to test a logistic regression paraphrase detection model on the STS benchmark (with Spearman Rank Correlation).
  * [ridge_classifier.py](ridge_classifier.py): Code to train a ridge regression paraphrase detection model.
  * [ridge_test.py](ridge_test.py): Code to test a ridge regression paraphrase detection model.
  * [cluster.py](cluster.py): Code to perform the agglomerative clustering on different datasets.

- Stale:
  * [model.py](model.py): Linear DNN architecture to learn paraphrase detection.
  * [main.py](main.py): Code to train a LinearDNN model to learn paraphrase detection.
  * [filter.py](filter.py): Code to filter on some intial results.
  * [utils.py](utils.py): Code for utility functions for Linear DNN training.
  * [visualize.py](visualize.py):Visualization on some intial results.


## Paraphrase detection:

* To start, we use SBERT embeddings on the MRPC dataset. Our training data is the MRPC train set + 5000 negative (contradictory)
pairs sampled at random from the SNLI dataset.
* For training, we use a logistic regression model, and the features for us are as follows:
  - We have input sentence pairs u and v
  - Model input is the vector ```(u,v,|u-v|,u\*v) and (v,u,|u-v|,u\*v)``` in order for the model to learn commutativity.
* The model was tested on the MRPC test dataset, with acceptable accuracy F1 scores for both classes.

## Agglomerative clustering:

* For this, first we take a test corpus and make pairs of sentences. This process becomes slow with increasing size of the corpus.
* For us, the test corpora were:
  1. Political corpus provided by Elliott.
  2. BBC speeches
  3. Trump Tweets
  4. Custom labelled provided by Dominik
  5. Extracted summaries (using MemSum) on Presidential Speeches 

## Status per corpora: 

First we use our paraphrase model to compute paraphrase probabilties pairwise for all the sentences in the corpora.

  1. The political corpora was too big to efficiently make all sentence pairs. There were around <b> 1150000 </b> sentences, which gives <b> 13225000000 </b> pairs. This
  doesn't scale, so we had this trick of computing the cosine similarity very fast (using efficient matrix multiplication) and filter pairs with low cosine 
  similarity. Then we compute the paraphrase probablities on the remaining pairs. The downside is this took an huge amount of memory, so we didn't proceed 
  computing the paraphrase probabilites.
  2. For the BBC corpus, we did both. We compute the pairwise sentence probabilites after tokenization. We make pairs in this case (which is slow but takes less memory) and also use the cosine similarity based matrix filtering (which is fast but takes more memory. 
  3. For the Trump tweet, we do not tokenize (as tweets don't work well with tokenization, due to not following grammatical structure). We compute the paraphrase probabilties for all pairs for this corpus.
  4. For the custom labelled corpus, we make pairs of all the sentences and compute the paraphrase probabilites. In the clustering process, we assign the cluster's majority label to all, and then we calculate the accuracy using this approach as a sanity check to see how good our clustering method is (<b> accurarcy </b> calculated using this method was <b>81% </b>).
  6. For the summaries extracted from Presidential speeches, we tokenize and compute the paraphrase probabilites on all pairs of sentences.

## Actual clustering mechansim:

  1. For this, we first initialise a similarity matrix over all possible sentence pairs, using the pairwise paraphrase probabilites. Then we create a distance matrix by subtracting this from the all ones matrix of the same shape.

  2. For the actual agglomerative clustering, we use this pre-computed distance matrix to fit on the model. After this, we get a linkage matrix from the model we fit with our distance matrix, and then we get the clusters from this linkage matrix.
  
  3. We then merge smaller clusters to larger ones, based on a depth factor which is up to us (we decided on <b> 0.45 </b>).

## Side note:

We also evaluated the paraphrase detection model on the STS dataset, which returned results which were acceptable. 

- correlation with yes prob: <b> SpearmanrResult(correlation=0.7635264850015682, pvalue=5.255517463444269e-49) </b>
- correlation with no prob: <b> SpearmanrResult(correlation=-0.7635264850015682, pvalue=5.255517463444269e-49) </b>
