import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np
from tensorflow.python.eager.context import num_gpus
import csv

walk_length = 150
num_walks = 20
vec_size = 64

author_embeddings = np.load('embeddings_array/author_embeddings_{}_{}_{}.npy'.format(walk_length, num_walks, vec_size))
author_ids = np.load('embeddings_array/author_ids_{}_{}_{}.npy'.format(walk_length, num_walks, vec_size))
paper_embeddings = np.load('embeddings_array/paper_embeddings_{}_{}_{}.npy'.format(walk_length, num_walks, vec_size))
paper_ids = np.load('embeddings_array/paper_ids_{}_{}_{}.npy'.format(walk_length, num_walks, vec_size))

transform = TSNE

'''trans = transform(n_components=2)
author_embeddings_2d = trans.fit_transform(author_embeddings)

node_colors = [1 for i in range(len(author_embeddings))]'''
paper_colors = [0 for i in range(len(paper_embeddings))]
with open('data/author_paper_all_with_year.csv') as f:
    reader = csv.reader(f)
    head = next(reader)
    data = list(reader)
    for row in data:
        paper_colors[int(row[1])] = int(row[2])-2015
print(paper_colors)
trans = transform(n_components=2)
paper_embeddings_2d = trans.fit_transform(paper_embeddings)



plt.figure(figsize=(20, 16))
plt.axes().set(aspect="equal")
plt.scatter(paper_embeddings_2d[:, 0], paper_embeddings_2d[:, 1], c=paper_colors, alpha=0.3)
plt.title("{} visualization of node embeddings".format(transform.__name__))
plt.savefig('paper3.jpg')