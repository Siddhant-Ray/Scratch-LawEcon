#!/usr/bin/env python
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import relatio
from relatio.utils import split_into_sentences
from relatio.wrappers import run_srl

# Path for files to be annotated
path_to_data_sets = (
    "/cluster/work/lawecon/Projects/Ash_Galletta_Widmer/data/scrapes_clean"
)
list_of_files = sorted(os.listdir(path_to_data_sets))

path_to_save = "/cluster/work/lawecon/Projects/annot_data_Ash_Widmer/daily_data"

semantics_dict = {}

# print(list_of_files)

for file_name in list_of_files:

    # only .csv files matter
    if file_name.endswith(".csv"):
        print(file_name)
        absolute_file_path = path_to_data_sets + "/" + file_name
        original_df = pd.read_csv(absolute_file_path)
        # print(df.head())
        # print(df['paragraph'])
        print(len(original_df.index))

        # make a doc_ID for the file in form of the date without - eg. 20050101
        document_ID = file_name.replace(".csv", "").split("-")
        document_ID = "".join(document_ID)

        # Copy of IDs to match the number of rows in the csv
        # Add the row number to map the paragraph_ID
        list_of_ID_for_df = [
            document_ID + "-" + str(i) for i in range(len(original_df.index))
        ]

        id_df = pd.DataFrame({"id": list_of_ID_for_df})
        # print(id_df.head())
        # print(len(id_df.index))

        doc_df = original_df["paragraph"]
        # print(doc_df.head())
        # print(len(doc_df.index))

        data = [id_df["id"], doc_df]
        headers = ["id", "doc"]

        # new data frame has two columns, IDs and doc(paragraph), as required
        # by the split_into_sentences_method
        new_data_frame = pd.concat(data, axis=1, keys=headers)
        print(new_data_frame.head())
        # print(len(new_data_frame.index))

        split_sentences = split_into_sentences(
            new_data_frame, progress_bar=True)
        """for i in range(5):
            print("document id: ", split_sentences[0][i])
            print("doc_in_sentences: ", split_sentences[1][i])
        """
        print(len(split_sentences[0]), len(split_sentences[1]))
        # break

        srl_results_per_paragraph = []
        for i in range(len(new_data_frame.index)):
            srl_results_per_paragraph.append([])

        print(len(srl_results_per_paragraph))
        # break

        cuda_str = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        cuda_device = int(cuda_str[0]) if cuda_str[0] else -1
        print(f"Using CUDA:{cuda_device}")

        srl_res = run_srl(
            # pre-trained model
            path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
            sentences=split_sentences[1],
            cuda_device=cuda_device,
            progress_bar=False,
        )

        print(len(srl_res))
        # break

        count_tracker = 0

        for number in range(len(split_sentences[1])):
            para_id = split_sentences[0][number].split("-")[1]
            position = int(para_id)

            srl_results_per_paragraph[position].append(srl_res[number])

        print(len(srl_results_per_paragraph))
        print(srl_results_per_paragraph[0])
        print()
        print(srl_results_per_paragraph[1])
        print()
        # break

        """for index, value in new_data_frame.iterrows():

            temp_df = pd.DataFrame([value])
            #temp_df = new_data_frame
            split_sentences = split_into_sentences(temp_df, progress_bar=False)
            #breakpoint()
            srl_res = run_srl(
            path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
            sentences=split_sentences[1],
            cuda_device=cuda_device,
            progress_bar=False)
            #print(len(split_sentences[1]))
            pd.set_option('display.max_colwidth', -1)
            print(value['doc'])
            srl_results_per_paragraph.append(srl_res)
            break

        print(len(srl_results_per_paragraph))
        print(len(srl_res))
        print(srl_results_per_paragraph)"""

        srl_dataframe = original_df
        srl_dataframe["srl_resutls"] = srl_results_per_paragraph

        file_name_to_save = path_to_save + "/" + file_name
        srl_dataframe.to_csv(file_name_to_save, index=False)

        # import time; time.sleep(5 * 60)
        break
