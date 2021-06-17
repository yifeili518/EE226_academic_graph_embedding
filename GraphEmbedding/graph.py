import pandas as pd
import itertools

def get_graph():
    author_paper = pd.read_csv('data/author_paper_all_with_year.csv')
    author = pd.DataFrame(index=author_paper['author_id'].values)
    author_prefix = author.set_index(
        "a" + author.index.astype(str)
    )
    author_prefix_list = author_prefix.index.tolist()

    paper = pd.DataFrame(index=author_paper['paper_id'].values)
    paper_prefix = paper.set_index(
        "p" + paper.index.astype(str)
    )
    paper_prefix_list = paper_prefix.index.tolist()

    paper_reference = pd.read_csv('data/paper_reference.csv')
    paper1 = pd.DataFrame(index=paper_reference['paper_id'].values)
    paper1_prefix = paper1.set_index(
        "p" + paper1.index.astype(str)
    )
    paper1_prefix_list = paper1_prefix.index.tolist()

    reference = pd.DataFrame(index=paper_reference['reference_id'].values)
    reference_prefix = reference.set_index(
        "p" + reference.index.astype(str)
    )
    reference_prefix_list = reference_prefix.index.tolist()

    author_paper = pd.read_csv('data/author_paper_all_with_year.csv')
    print(author_paper.head())

    a_id = author_paper['author_id'].tolist()
    p_id = author_paper['paper_id'].tolist()
    author_author1 = []
    author_author2 = []
    mat = []

    for j in range(len(list(set(p_id)))):
        tmp = []
        for i in range(len(p_id)):
            if p_id[i] == j:
                tmp.append(a_id[i])
        cc = list(itertools.combinations(tmp, 2))
        for k in range(len(cc)):
            author_author1.append('a'+str(cc[k][0]))
            author_author2.append('a'+str(cc[k][1]))

    source = author_prefix_list + paper1_prefix_list + author_author1
    target = paper_prefix_list + reference_prefix_list + author_author2

    edges = pd.DataFrame(
        {"source":source, "target":target}
    )

    author_index = pd.DataFrame(author_paper["author_id"].values)
    N = author_index.max()
    author_i = range(N[0] + 1)
    author_i = pd.DataFrame(index=author_i)
    author_index_prefix = author_i.set_index(
        "a" + author_i.index.astype(str)
    )

    paper_index = pd.DataFrame(author_paper["paper_id"].values)
    M = paper_index.max()
    paper_i = range(M[0] + 1)
    paper_i = pd.DataFrame(index=paper_i)
    paper_index_prefix = paper_i.set_index(
        "p" + paper_i.index.astype(str)
    )

    from stellargraph import StellarGraph
    g = StellarGraph({"author": author_index_prefix, "paper":paper_index_prefix}, edges)
    return g
    print(g.info())