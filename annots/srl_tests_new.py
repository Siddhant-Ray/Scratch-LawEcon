#!/usr/bin/env python

import pandas as pd
import os
import relatio
import numpy as np

from relatio.wrappers import run_srl
from relatio.utils import split_into_sentences

# Path for files to be annotated
path_to_data_sets = "/cluster/work/lawecon/Projects/Ash_Galletta_Widmer/data/scrapes_clean"
list_of_files = sorted(os.listdir(path_to_data_sets))

path_to_save = "/cluster/work/lawecond/Projects/annot_data_Ash_Widmer/daily_data"

semantics_dict = {}

#print(list_of_files)

for file_name in list_of_files:

    # only .csv files matter
    if file_name.endswith(".csv"):
        print(file_name)
        absolute_file_path = path_to_data_sets + "/" + file_name
        original_df = pd.read_csv(absolute_file_path)
        #print(df.head())
        #print(df['paragraph'])
        print(len(original_df.index))

        # make a doc_ID for the file in form of the date without - eg. 20050101
        document_ID = file_name.replace(".csv","").split("-")
        document_ID = "".join(document_ID)
        
        # Copy of IDs to match the number of rows in the csv
        list_of_ID_for_df = [document_ID for i in range(len(original_df.index))]
        
        id_df = pd.DataFrame({'id':list_of_ID_for_df})
        #print(id_df.head())
        #print(len(id_df.index))

        doc_df = original_df['paragraph']
        #print(doc_df.head())
        #print(len(doc_df.index))

        data = [id_df['id'], doc_df]
        headers = ['id', 'doc']

        # new data frame has two columns, IDs and doc(paragraph), as required
        # by the split_into_sentences_method
        new_data_frame = pd.concat(data, axis=1, keys=headers)
        #print(new_data_frame.head())
        #print(len(new_data_frame.index))

        '''split_sentences = split_into_sentences(new_data_frame, progress_bar=True)
        for i in range(5):
            print("document id: ", split_sentences[0][i])
            print("doc_in_sentences: ", split_sentences[1][i])'''

        srl_results_per_paragraph = []
        
        for index, value in new_data_frame.iterrows():
            temp_df = pd.DataFrame([value])
            split_sentences = split_into_sentences(temp_df, progress_bar=False)
            srl_res = run_srl(
            path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
            sentences=split_sentences[1],
            cuda_device=-1,
            progress_bar=False)
            # print(len(split_sentences[1]))

            srl_results_per_paragraph.append(srl_res)
            
        print(len(srl_results_per_paragraph))
        #print(srl_results_per_paragraph)

        srl_dataframe = original_df 
        srl_dataframe['srl_resutls'] = srl_results_per_paragraph
        
        file_name_to_save = path_to_save + "/" + file_name
        srl_dataframe.to_csv(file_name_to_save, index = False)






        '''srl_res = run_srl(
        path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
        sentences=split_sentences[1],
        cuda_device=-1,
        progress_bar=True,
        )





        print(srl_res[0])

        list_of_verbs = []
        #list_of_args = []
        list_of_words = []

        for item in srl_res:

            list_of_verbs.append(item['verbs'])
            #list_of_args.append(item['verbs'][1])
            list_of_words.append(item['words'])

        print(len(list_of_verbs))
        #print(len(list_of_args))
        print(len(list_of_words))

        id_dataframe = pd.DataFrame(split_sentences[0])
        sent_dataframe = pd.DataFrame(split_sentences[1])
        verbs_dataframe = pd.DataFrame(list_of_verbs)
        words_dataframe = pd.DataFrame(list_of_words)
        df_values = [id_dataframe, sent_dataframe, verbs_dataframe, words_dataframe]
        df_keys = ['id', 'sentence', 'verb_action', 'words']

        final_dataframe = pd.concat(df_values, axis=1, keys = df_keys)
        print(final_dataframe.head())

        final_dataframe.to_csv("test.csv", index = False)'''
        break



