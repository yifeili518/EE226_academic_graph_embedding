import pandas as pd
import numpy as np

author_citation = pd.read_csv('author_citation_shuangxiang.csv')
x = np.unique(author_citation['author_id'].values)
print(len(x))
coauthor = pd.read_csv('coauthor.csv')
y = np.unique(coauthor['author_id'].values)
print(len(y))
co_citation = pd.read_csv('co_citation.csv')
z = np.unique(co_citation['author_id'].values)
print(len(z))