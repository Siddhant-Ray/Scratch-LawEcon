#!/usr/bin/env python

import pandas as pd
import os, sys
import relatio
import numpy as np
from pathlib import Path

from dateutil import parser

from relatio.wrappers import run_srl
from relatio.utils import split_into_sentences

# Path for files to be annotated
path_to_data_sets = "/cluster/home/sidray/work/Ash_Galletta_Widmer/data/scrapes_since_1980"
list_of_files = sorted(os.listdir(path_to_data_sets))

path_to_save = "/cluster/work/lawecon/Projects/annot_data_Ash_Widmer/non_daily_data"

semantics_dict = {}


file_path = sys.argv[1]
file_name = file_path.split("/")[-1]
print(file_path, file_name)
# exit()

# Testing for one particular file
# file_name = "2005-01-02.csv"
# file_path = path_to_data_sets + "/" + file_name

# only .csv files matter
if file_name.endswith(".csv"):
    print(file_name)
    absolute_file_path = file_path
    original_df = pd.read_csv(absolute_file_path)
    print(original_df.head())

    folder_name = file_name.split(".")[0][-4:]
    print(folder_name)
    temp_path = path_to_data_sets + "/" +  folder_name
    print(temp_path)

    if Path(temp_path).is_dir():
            print("yes")
    else:
        print("no")
        new_path = os.path.join(path_to_data_sets, folder_name)
        os.mkdir(new_path)
        print("New path is: {}".format(new_path))
    
    new_path = temp_path
    # print(Path(path_to_data_sets + folder_name).is_dir())
    dfs = dict(tuple(original_df.groupby('date')))

    for i, df in dfs.items():
        date_text = df['date'].iloc[0]
        date = parser.parse(date_text)
        date = date.strftime('%d-%m-%Y')
        print(date)
        
        df.to_csv(new_path + "/" + date + ".csv", index = False)
        

   



