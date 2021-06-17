import pandas as pd
import os
 
 
data = pd.read_csv('../data/authors_to_themselves.csv', encoding='utf-8')
with open('self_edges.txt','a+', encoding='utf-8') as f:
    for line in data.values:
        f.write((str(line[0])+'\t'+str(line[1])+'\n'))
