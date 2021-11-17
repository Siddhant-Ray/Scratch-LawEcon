#!usr/bin/env python

import sys
import pandas as pd
print(sys.argv[1])
_file = pd.read_csv(sys.argv[1])
print(_file.head())

