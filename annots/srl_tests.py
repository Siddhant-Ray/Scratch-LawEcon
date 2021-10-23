#!/usr/bin/env python

import pandas as pd
import os
import relatio

from relatio.wrappers import run_srl
from relatio.utils import split_into_sentences

path_to_data_sets = "/cluster/work/lawecon/Projects/Ash_Galletta_Widmer/data/scrapes_since_1980"
list_of_files = sorted(os.listdir(path_to_data_sets))

semantics_dict = {}

print(list_of_files)


for file_name in list_of_files:
    # Keep a counter for the entry number in the DataFrame object
    count = 0


	if file_name.endswith(".csv"):
		print(file_name)
		absolute_file_path = path_to_data_sets + "/" + file_name
		df = pd.read_csv(absolute_file_path)

	print(df.head())
	print(df['text'])

		for index, row in df.iterrows():
			print(row['text'], row['title'])

			srl_result = run_srl(
				path="https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz", # pre-trained model
				sentences = row['text'],
				#cuda_device = -1,
				progress_bar = True,
			)

			if file_name not in semantics_dict.keys():

				semantics_dict[filename] = {}

			semantics_dict[filename][row['title']] = srl_result
			count += 1

			break

	print(srl_result)
	print(semantics_dict)

	break


print("test string")
