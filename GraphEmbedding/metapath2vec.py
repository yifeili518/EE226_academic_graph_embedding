import stellargraph as sg
print(sg.__version__)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import datasets, StellarGraph
from IPython.display import display, HTML
import os

from link_pred1 import link_prediction
from node_class import node_classification
from paper_class import paper_classification
from graph import get_graph


class Academic():
    def __init__(self):
        self.name = "Academic"
        self.expected_files = ["data/paper_reference.csv",
        "data/authorship.csv",
        "data/papers.csv",
        "data/authors.csv",
        "data/coauthor_single.csv"]

    def load(self):
        paper_reference, authorship, papers, authors, coauthor= [
            name for name in self.expected_files
        ]
        authors = pd.read_csv(authors, header=None)
        papers = pd.read_csv(papers, header=None)
        paper_reference = pd.read_csv(paper_reference, header=None, names=["source", "target"])
        authorship = pd.read_csv(authorship, header=None, names=["source", "target"])
        coauthor = pd.read_csv(coauthor, header=None, names=["source", "target"])
        # The dataset uses integers for node ids. However, the integers from 1 to 39 are used as IDs
        # for both users and groups. This is disambiguated by converting everything to strings and
        # prepending u to user IDs, and g to group IDs.
        def a(authors):
            return "a" + authors.astype(str)

        def p(papers):
            return "p" + papers.astype(str)

        # nodes:
        authors = a(authors)
        papers = p(papers)

        # node IDs in each edge:
        paper_reference = p(paper_reference)
        coauthor = a(coauthor)
        authorship["source"] = a(authorship["source"])
        authorship["target"] = p(authorship["target"])
        
        # arrange the DataFrame indices appropriately: nodes use their node IDs, which have
        # been made distinct above, and the group edges have IDs after the other edges
        authors.set_index(0, inplace=True)
        papers.set_index(0, inplace=True)

        start = len(paper_reference)
        authorship.index = range(start, start + len(authorship))
        coauthor.index = range(start+len(authorship), start+len(authorship)+len(coauthor))

        return StellarGraph(
            nodes={"author": authors, "paper": papers},
            edges={"paper_reference": paper_reference, "authorship": authorship, "coauthor":coauthor},
        )

walk_length = 120  # maximum length of a random walk to use throughout this notebook
n = 10 #num of walks for each node
vec_size = 256

if not os.path.exists('embeddings_array/author_embeddings_{}_{}_{}.npy'.format(walk_length,n, vec_size)):
    dataset = Academic()
    g = dataset.load()
    print(g.info())


    # specify the metapath schemas as a list of lists of node types.
    metapaths = [
        ["author", "paper", "author"],
        ["author", "paper", "paper", "author"],
        ["paper", "paper"],
        ["author", "author"]
    ]

    from stellargraph.data import UniformRandomMetaPathWalk

    # Create the random walker

    rw = UniformRandomMetaPathWalk(g)

    walks = rw.run(
        nodes=list(g.nodes()),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=n,  # number of random walks per root node
        metapaths=metapaths,  # the metapaths
    )

    print("Number of random walks: {}".format(len(walks)))


    from gensim.models import Word2Vec

    model = Word2Vec(walks, vector_size=vec_size, window=5, min_count=0, sg=1, workers=2, epochs=1)

    embeddings = model.wv.vectors  # 128-dimensional vector for each node in the graph
    print(embeddings.shape)
    node_ids = model.wv.index_to_key
    node_types = [g.node_type(node_id) for node_id in node_ids]
    print(type(embeddings))
    print(embeddings[0])
    print(type(embeddings[0]))
    author_embeddings = []
    author_ids = []
    paper_embeddings = []
    paper_ids = []
    for i in range(len(embeddings)):
        if node_types[i] == 'author':
            author_embeddings.append(embeddings[i])
            author_ids.append(node_ids[i][1:]) #去掉前缀author
        if node_types[i] == 'paper':
            paper_embeddings.append(embeddings[i])
            paper_ids.append(node_ids[i][1:]) #去掉前缀paper

    print(len(author_embeddings))
    print(len(paper_embeddings))

    np.save('embeddings_array/author_embeddings_{}_{}_{}.npy'.format(walk_length, n,vec_size),np.array(author_embeddings))
    np.save('embeddings_array/author_ids_{}_{}_{}.npy'.format(walk_length, n, vec_size), np.array(author_ids))
    np.save('embeddings_array/paper_embeddings_{}_{}_{}.npy'.format(walk_length, n, vec_size),np.array(paper_embeddings))
    np.save('embeddings_array/paper_ids_{}_{}_{}.npy'.format(walk_length, n, vec_size), np.array(paper_ids))
else:
    author_embeddings = np.load('embeddings_array/author_embeddings_{}_{}_{}.npy'.format(walk_length,n, vec_size)).tolist()
    author_ids = np.load('embeddings_array/author_ids_{}_{}_{}.npy'.format(walk_length, n, vec_size)).tolist()
    paper_embeddings = np.load('embeddings_array/paper_embeddings_{}_{}_{}.npy'.format(walk_length,n, vec_size)).tolist()
    paper_ids = np.load('embeddings_array/paper_ids_{}_{}_{}.npy'.format(walk_length, n, vec_size)).tolist()

print(len(author_embeddings))
print(len(author_ids))
print(type(author_ids[0]))
print(author_ids)

link_prediction(author_embeddings, author_ids,walk_length, n, vec_size)
#node_classification(author_embeddings, author_ids, walk_length, n, vec_size)
#paper_classification(paper_embeddings, paper_ids, walk_length, n)
