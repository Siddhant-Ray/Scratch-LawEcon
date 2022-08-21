from __future__ import annotations

import json
import math
import pickle
import random

from sentence_transformers import SentenceTransformer

file_path_1 = "paraphrase/MSRParaphraseCorpus/msr_paraphrase_train.txt"
file_path_2 = "paraphrase/multinli_1.0/multinli_1.0_train.jsonl"
file1 = open(file_path_1)
lines_of_sentences = file1.readlines()

# list to store values
labeled_pairs_list = []

for value in lines_of_sentences:
    # split on "\t"
    data = value.split("\t")
    label = data[0]

    # list of pairs of sentences needed
    sent_pairs = [data[3].strip("\n"), data[4].strip("\n")]
    # labeled pairs
    labeled_pairs_list.append([label, sent_pairs])

# remove item 1 with headings
labeled_pairs_list.pop(0)
count_1 = 0
count_0 = 0

for item in labeled_pairs_list:
    if int(item[0]) == 1:
        count_1 += 1
    else:
        count_0 += 1

# print(count_1,count_0)

with open(file_path_2, "r") as json_file:
    json_list = list(json_file)

nli_contr_count = 0
nli_pairs_list = []
for json_str in json_list:
    result = json.loads(json_str)
    if result["annotator_labels"] == ["contradiction"]:
        label = str(0)
        sent1 = result["sentence1"]
        sent2 = result["sentence2"]
        sent_pairs = [sent1, sent2]
        nli_contr_count += 1
        nli_pairs_list.append([label, sent_pairs])


subset = random.sample(nli_pairs_list, 5000)
# print(subset[0:5])
# print(len(subset))
list_of_MRPC_pairs = labeled_pairs_list
print(len(list_of_MRPC_pairs))
# print(list_of_MRPC_pairs[0:5])


final_list_of_pairs = random.sample(
    labeled_pairs_list + subset, len(labeled_pairs_list + subset)
)
print(len(final_list_of_pairs))
# print(final_list_of_pairs[0:5])

# print(len(nli_pairs_list))
# print(nli_pairs_list[0:5])

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
train_labels = [int(item[0]) for item in final_list_of_pairs]
sentence_1_list = [item[1][0] for item in final_list_of_pairs]
sentence_2_list = [item[1][1] for item in final_list_of_pairs]

print(len(train_labels), len(sentence_1_list), len(sentence_2_list))
"""embeddings_1 = model.encode(sentence_1_list)
embeddings_2 = model.encode(sentence_2_list)"""

"""with open('paraphrase/data/embeddings_1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': sentence_1_list, 'embeddings': embeddings_1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/embeddings_2.pkl', "wb") as fOut2:
    pickle.dump({'sentences': sentence_2_list, 'embeddings': embeddings_2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/labels.pkl', "wb") as fOut3:
    pickle.dump({'labels': train_labels}, fOut3, protocol=pickle.HIGHEST_PROTOCOL)"""


# Embeddings for only MPRC dataset
train_labels_MPRC = [int(item[0]) for item in list_of_MRPC_pairs]
MPRC_list_1 = [item[1][0] for item in list_of_MRPC_pairs]
MPRC_list_2 = [item[1][1] for item in list_of_MRPC_pairs]

print(len(train_labels_MPRC), len(MPRC_list_1), len(MPRC_list_2))

"""mprc_embeddings_1 = model.encode(MPRC_list_1)
mprc_embeddings_2 = model.encode(MPRC_list_2)

with open('paraphrase/data/mprc_embeddings_1.pkl', "wb") as fOut_mprc1:
    pickle.dump({'sentences': MPRC_list_1, 'embeddings': mprc_embeddings_1}, fOut_mprc1, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/mprc_embeddings_2.pkl', "wb") as fOut_mprc2:
    pickle.dump({'sentences': MPRC_list_2, 'embeddings': mprc_embeddings_2}, fOut_mprc2, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/mprc_labels.pkl', "wb") as fOut_mprc3:
    pickle.dump({'labels': train_labels_MPRC}, fOut_mprc3, protocol=pickle.HIGHEST_PROTOCOL)"""


# Testing vectors using MPRC test file only:

file_path_test = "paraphrase/MSRParaphraseCorpus/msr_paraphrase_test.txt"

file_test = open(file_path_test)
lines_of_sentences = file_test.readlines()

# list to store values
test_pairs_list = []

for value in lines_of_sentences:
    # split on "\t"
    data = value.split("\t")
    label = data[0]

    # list of pairs of sentences needed
    sent_pairs = [data[3].strip("\n"), data[4].strip("\n")]
    # labeled pairs
    test_pairs_list.append([label, sent_pairs])

# remove item 1 with headings
test_pairs_list.pop(0)
count_1 = 0
count_0 = 0

for item in test_pairs_list:
    if int(item[0]) == 1:
        count_1 += 1
    else:
        count_0 += 1

# print(count_1,count_0)

list_of_test_pairs = test_pairs_list

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
test_labels = [int(item[0]) for item in list_of_test_pairs]
test_1_list = [item[1][0] for item in list_of_test_pairs]
test_2_list = [item[1][1] for item in list_of_test_pairs]

print(len(test_labels), len(test_1_list), len(test_2_list))

"""test_embeddings1 = model.encode(test_1_list)
test_embeddings2 = model.encode(test_2_list)

with open('paraphrase/data/test_embeddings_1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': test_1_list, 'embeddings': test_embeddings1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/test_embeddings_2.pkl', "wb") as fOut2:
    pickle.dump({'sentences': test_2_list, 'embeddings': test_embeddings2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/test_labels.pkl', "wb") as fOut3:
    pickle.dump({'labels': test_labels}, fOut3, protocol=pickle.HIGHEST_PROTOCOL)"""


file_path_3 = "paraphrase/paws_corpus/test.tsv"

file3 = open(file_path_3)
lines_of_sentences_paws = file3.readlines()

# list to store values
labeled_pairs_list_paws = []

for value in lines_of_sentences_paws:
    # split on "\t"
    data = value.split("\t")
    label = data[3].strip("\n")

    # list of pairs of sentences needed
    sent_pairs = [data[1].strip("\n"), data[2].strip("\n")]
    # labeled pairs
    labeled_pairs_list_paws.append([label, sent_pairs])

labeled_pairs_list_paws.pop(0)
# print(labeled_pairs_list_paws[0])
# print(labeled_pairs_list_paws[1])

count_1 = 0
count_0 = 0

for item in labeled_pairs_list_paws:
    if int(item[0]) == 1:
        count_1 += 1
    else:
        count_0 += 1

# print(count_1, count_0)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
test_labels = [int(item[0]) for item in labeled_pairs_list_paws]
test_1_list = [item[1][0] for item in labeled_pairs_list_paws]
test_2_list = [item[1][1] for item in labeled_pairs_list_paws]

print(len(test_labels), len(test_1_list), len(test_2_list))

"""test_embeddings_paws1 = model.encode(test_1_list)
test_embeddings_paws2 = model.encode(test_2_list)"""

"""with open('paraphrase/data/test_embeddings_paws1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': test_1_list, 'embeddings': test_embeddings_paws1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/test_embeddings_paws2.pkl', "wb") as fOut2:
    pickle.dump({'sentences': test_2_list, 'embeddings': test_embeddings_paws2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/test_labels_paws.pkl', "wb") as fOut3:
    pickle.dump({'labels': test_labels}, fOut3, protocol=pickle.HIGHEST_PROTOCOL)"""


# Also try to train on PAWS dataset, generate PAWS train embeddings

file_path_4 = "paraphrase/paws_corpus/train.tsv"

file4 = open(file_path_4)
lines_of_sentences_paws_train = file4.readlines()

# list to store values
labeled_pairs_list_paws_train = []

for value in lines_of_sentences_paws_train:
    # split on "\t"
    data = value.split("\t")
    label = data[3].strip("\n")

    # list of pairs of sentences needed
    sent_pairs = [data[1].strip("\n"), data[2].strip("\n")]
    # labeled pairs
    labeled_pairs_list_paws_train.append([label, sent_pairs])

labeled_pairs_list_paws_train.pop(0)
# print(labeled_pairs_list_paws_train[0])
# print(labeled_pairs_list_paws_train[1])

count_1 = 0
count_0 = 0

for item in labeled_pairs_list_paws_train:
    if int(item[0]) == 1:
        count_1 += 1
    else:
        count_0 += 1

# print(count_1, count_0)

model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
train_labels_paws = [int(item[0]) for item in labeled_pairs_list_paws_train]
train_1_list = [item[1][0] for item in labeled_pairs_list_paws_train]
train_2_list = [item[1][1] for item in labeled_pairs_list_paws_train]

print(len(train_labels_paws), len(train_1_list), len(train_2_list))

"""train_embeddings_paws1 = model.encode(train_1_list)
train_embeddings_paws2 = model.encode(train_2_list)

with open('paraphrase/data/train_embeddings_paws1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': train_1_list, 'embeddings': train_embeddings_paws1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/train_embeddings_paws2.pkl', "wb") as fOut2:
    pickle.dump({'sentences': train_2_list, 'embeddings': train_embeddings_paws2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)

with open('paraphrase/data/train_labels_paws.pkl', "wb") as fOut3:
    pickle.dump({'labels': train_labels_paws}, fOut3, protocol=pickle.HIGHEST_PROTOCOL)"""
