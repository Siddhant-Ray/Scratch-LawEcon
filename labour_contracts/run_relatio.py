import pandas as pd
import os, sys
import relatio
import numpy as np
import pickle

from relatio.wrappers import run_srl
from relatio.utils import split_into_sentences
from relatio.wrappers import build_narrative_model, get_narratives
from relatio.graphs import build_graph, draw_graph

import nltk
nltk.download('averaged_perceptron_tagger')

import spacy
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

## Flags 
split_sents = False
run_srl_method = False
run_narrative = False
gen_narratives = False
analyse_narratives = False
plot_graph = True

data_path = "labour_contracts/data/relatio_formatted.csv"

## load data 
data_frame = pd.read_csv(data_path)
print(data_frame.head(10))
print(data_frame.shape)

new_cols = ["id", "doc"]
new_frame = data_frame[["id", "text"]]
new_frame.columns = new_cols
new_frame.fillna('', inplace=True)

print(new_frame.head(10), new_frame.shape)

# Choose 500 articles
new_short_frame = new_frame.head(500)

# SPLIT into sentences
if split_sents:

    split_sentences = split_into_sentences(new_short_frame, progress_bar=True)

    with open('labour_contracts/data/sentences.pkl', "wb") as sent_file:
        pickle.dump({'id': split_sentences[0], 'sentences': split_sentences[1]}, sent_file, protocol=pickle.HIGHEST_PROTOCOL)

cuda_str = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
cuda_device = int(cuda_str[0]) if cuda_str[0] else -1
print(f"Using CUDA:{cuda_device}")

# USE GPU, RUN SRL method
if run_srl_method:
    assert(cuda_device!=-1)

    with open("labour_contracts/data/sentences.pkl", 'rb') as f:
        sentences_file = pickle.load(f)

    ids, sentences = sentences_file["id"], sentences_file["sentences"]
    print(len(ids), len(sentences))

    srl_res = run_srl(
                path = "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
                sentences = sentences,
                cuda_device = cuda_device,
                progress_bar = True)

    print(len(srl_res))

    with open('labour_contracts/data/srl.pkl', "wb") as sent_file:
        pickle.dump({'srl' : srl_res , 'sentences': sentences}, sent_file, protocol=pickle.HIGHEST_PROTOCOL)

# BUILD narrative model
if run_narrative:

    with open('labour_contracts/data/srl.pkl', 'rb') as f:
        file_load = pickle.load(f)

    srl_res = file_load['srl']
    sentences = file_load['sentences']

    print(srl_res[0:10])
    print(sentences[0:10])

    print(len(sentences), len(srl_res))

    narrative_model = build_narrative_model(
        srl_res=srl_res,
        sentences=sentences,
        roles_considered = [
        "ARG0",
        "B-V",
        "B-ARGM-NEG",
        "B-ARGM-MOD",
        "ARG1",
        "ARG2",
        ],
        embeddings_type="gensim_keyed_vectors", 
        embeddings_path="glove-wiki-gigaword-100",
        n_clusters=[[8]],
        roles_with_entities = [],
        top_n_entities=100,
        stop_words = spacy_stopwords,
        remove_n_letter_words = 1,
        progress_bar=True,
    )

    print(narrative_model['entities'].most_common()[:20])

    with open('labour_contracts/data/model.pkl', "wb") as sent_file:
        pickle.dump({'model' : narrative_model}, sent_file, protocol=pickle.HIGHEST_PROTOCOL)

# GENERATE narratives
if gen_narratives:

    with open('labour_contracts/data/model.pkl', 'rb') as f:
        file_load = pickle.load(f)
        narrative_model = file_load['model']

    with open('labour_contracts/data/srl.pkl', 'rb') as f:
        file_load = pickle.load(f)
        srl_res = file_load['srl']

    with open("labour_contracts/data/sentences.pkl", 'rb') as f:
        sentences_file = pickle.load(f)

    ids, sentences = sentences_file["id"], sentences_file["sentences"]
    print(len(ids), len(sentences), len(srl_res))
          
    final_statements = get_narratives(
            srl_res = srl_res,
            doc_index = ids, 
            narrative_model=narrative_model,
            n_clusters=[0],  
            progress_bar=True,
        )

    print(final_statements.columns)
    print(final_statements.head())

    final_statements.to_csv('labour_contracts/data/narratives.csv', index = False)

# ANALYSE sentiments (this is mainly done very similarly as the tutorial)    
if analyse_narratives:
    final_statements = pd.read_csv('labour_contracts/data/narratives.csv')
    print(final_statements.columns)

    # Entity coherence
    # Print most frequent phrases per entity

    # Pool ARG0, ARG1 and ARG2 together
    df1 = final_statements[['ARG0_lowdim', 'ARG0_highdim']]
    df1.rename(columns={'ARG0_lowdim': 'ARG', 'ARG0_highdim': 'ARG-RAW'}, inplace=True)

    df2 = final_statements[['ARG1_lowdim', 'ARG1_highdim']]
    df2.rename(columns={'ARG1_lowdim': 'ARG', 'ARG1_highdim': 'ARG-RAW'}, inplace=True)

    df3 = final_statements[['ARG2_lowdim', 'ARG2_highdim']]
    df3.rename(columns={'ARG2_lowdim': 'ARG', 'ARG2_highdim': 'ARG-RAW'}, inplace=True)

    df = df1.append(df2).reset_index(drop = True)
    df = df.append(df3).reset_index(drop = True)

    # Count semantic phrases
    df = df.groupby(['ARG', 'ARG-RAW']).size().reset_index()
    df.columns = ['ARG', 'ARG-RAW', 'count']

    # Drop empty semantic phrases
    df = df[df['ARG'] != ''] 

    # Rearrange the data
    df = df.groupby(['ARG']).apply(lambda x: x.sort_values(["count"], ascending = False))
    df = df.reset_index(drop= True)
    df = df.groupby(['ARG']).head(10)

    df['ARG-RAW'] = df['ARG-RAW'] + ' - ' + df['count'].astype(str)
    df['cluster_elements'] = df.groupby(['ARG'])['ARG-RAW'].transform(lambda x: ' | '.join(x))

    df = df.drop_duplicates(subset=['ARG'])

    df['cluster_elements'] = [', '.join(set(i.split(','))) for i in list(df['cluster_elements'])]

    print('Entities to inspect:', len(df))
    df = df[['ARG', 'cluster_elements']]

    for l in df.values.tolist():
        pass
        #print('entity: \n %s \n' % l[0])
        # print('most frequent phrases: \n %s \n' % l[1])

    
    # Low-dimensional vs. high-dimensional narrative statements
    # Replace negated verbs by "not-verb"

    ## In our data, this leads to many NANs but some values are quite good

    final_statements['B-V_lowdim_with_neg'] = np.where(final_statements['B-ARGM-NEG_lowdim'] == True, 
                                            final_statements['B-V_lowdim'],
                                            'not_' + final_statements['B-V_lowdim'])

    final_statements['B-V_highdim_with_neg'] = np.where(final_statements['B-ARGM-NEG_highdim'] == True, 
                                            final_statements['B-V_highdim'],
                                            'not_' + final_statements['B-V_lowdim']) 

    
    # Concatenate high-dimensional narratives (with text preprocessing but no clustering)

    final_statements['narrative_highdim'] = (final_statements['ARG0_highdim'] + ' ' + 
                                            final_statements['B-V_highdim_with_neg'] + ' ' +  
                                            final_statements['ARG1_highdim'])

    # Concatenate low-dimensional narratives (with clustering)

    final_statements['narrative_lowdim'] = (final_statements['ARG0_lowdim'] + ' ' + 
                                            final_statements['B-V_highdim_with_neg'] + ' ' + 
                                            final_statements['ARG1_lowdim'])

    # Focus on narratives with a ARG0-VERB-ARG1 structure (i.e. "complete narratives")

    indexNames = final_statements[(final_statements['ARG0_lowdim'] == '')|
                                (final_statements['ARG1_lowdim'] == '')|
                                (final_statements['B-V_lowdim_with_neg'] == '')].index

    complete_narratives = final_statements.drop(indexNames)

    print(complete_narratives.head())
    print(complete_narratives['narrative_highdim'].value_counts().head(10))
    print(complete_narratives['narrative_lowdim'].value_counts().head(10))

    ## Change from original , filter out the NaNs
    sample = complete_narratives[~((complete_narratives['narrative_highdim'].isnull()) & (complete_narratives['narrative_lowdim'].isnull()))].sample(10, random_state = 123).to_dict('records')

    with open("labour_contracts/data/sentences.pkl", 'rb') as f:
        sentences_file = pickle.load(f)

    ids, sentences = sentences_file["id"], sentences_file["sentences"]

    write_file = open('labour_contracts/data/high-low-dim-narratives.txt', 'w')
    for d in sample:
        print('Original sentence : \n %s \n' %sentences[d['sentence']])
        print('High-dimensional narrative: \n %s \n' %d['narrative_highdim'])
        print('Low-dimensional narrative: \n %s \n' %d['narrative_lowdim'])
        print('--------------------------------------------------- \n')

        write_file.write('Original sentence : \n %s \n' %sentences[d['sentence']])
        write_file.write('High-dimensional narrative: \n %s \n' %d['narrative_highdim'])
        write_file.write('Low-dimensional narrative: \n %s \n' %d['narrative_lowdim'])
        write_file.write('--------------------------------------------------- \n')

    write_file.close()
    complete_narratives.to_csv('labour_contracts/data/complete_narratives.csv', index = False)

# PLOT the directed graph
if plot_graph:

    complete_narratives = pd.read_csv('labour_contracts/data/complete_narratives.csv')
    print(complete_narratives.columns)
    temp = complete_narratives[["ARG0_lowdim", "ARG1_lowdim", "B-V_lowdim", "B-ARGM-MOD_highdim"]]
    temp.columns = ["ARG0", "ARG1", "B-V", "B-M"]
    temp = temp[(temp["ARG0"] != "") & (temp["ARG1"] != "") & (temp["B-V"] != "") & (temp["B-M"] != 0)]
    temp = temp.groupby(["ARG0", "ARG1", "B-V", "B-M"]).size().reset_index(name="weight")
    temp = temp.sort_values(by="weight", ascending=False).iloc[0:100]  # pick top 100 most frequent narratives
    temp = temp.to_dict(orient="records")

    print(temp)

    for l in temp:
        l["color"] = None

    G = build_graph(
        dict_edges=temp, dict_args={}, edge_size=None, node_size=2, prune_network=True
    )

    draw_graph(G, notebook=True, width="50000px", height="50000px", output_filename="labour_contracts/data/final_graph.html")
