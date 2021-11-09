import json 
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

    #list of pairs of sentences needed  
    sent_pairs = [data[3].strip("\n"), data[4].strip("\n")]
    #labeled pairs 
    labeled_pairs_list.append([label, sent_pairs])

# remove item 1 with headings
labeled_pairs_list.pop(0)
count_1 = 0
count_0 = 0 

for item in labeled_pairs_list:
    if int(item[0])== 1:
        count_1 += 1 
    else:
        count_0 += 1

#print(count_1,count_0)

with open(file_path_2, 'r') as json_file:
    json_list = list(json_file)

nli_contr_count = 0 
nli_pairs_list = []
for json_str in json_list:
    result = json.loads(json_str)
    if result['annotator_labels']==['contradiction']:
        label = str(0)
        sent1 = result['sentence1']
        sent2 = result['sentence2']
        sent_pairs = [sent1, sent2]
        nli_contr_count+=1
        nli_pairs_list.append([label, sent_pairs])

subset = random.sample(nli_pairs_list, 10000)
#print(subset[0:5])
#print(len(subset))

final_list_of_pairs = random.sample(labeled_pairs_list + subset, len(labeled_pairs_list + subset))
print(len(final_list_of_pairs))
print(final_list_of_pairs[0:5])

#print(len(nli_pairs_list))
#print(nli_pairs_list[0:5])

model = SentenceTransformer('all-MiniLM-L6-v2', device = 'cuda')    
train_labels = [int(item[0]) for item in final_list_of_pairs] 
sentence_1_list = [item[1][0] for item in final_list_of_pairs]
sentence_2_list = [item[1][1] for item in final_list_of_pairs]

#print(len(train_labels),len(sentence_1_list),len(sentence_2_list))
embeddings_1 = model.encode(sentence_1_list)
embeddings_2 = model.encode(sentence_2_list)

with open('embeddings_1.pkl', "wb") as fOut1:
    pickle.dump({'sentences': sentences_1_list, 'embeddings': embeddings_1}, fOut1, protocol=pickle.HIGHEST_PROTOCOL)

with open('embeddings_2.pkl', "wb") as fOut2:
    pickle.dump({'sentences': sentences_2_list, 'embeddings': embeddings_2}, fOut2, protocol=pickle.HIGHEST_PROTOCOL)
    


 
