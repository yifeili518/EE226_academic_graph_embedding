import csv
import numpy as np
import random

def read_pairs_toPred(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        head = next(reader)
        data = list(reader)
        paris = []
        for row in data:
            pair = row[1].split()
            paris.append(pair)
    
    return paris


def store_result(filename, score):
    with open(filename,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for i, prob in enumerate(score):
            i = str(i)
            prob = str(prob)
            writer.writerow([i, prob])

def cosine_similarity(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def tanimoto_coefficient(p_vec, q_vec):    
    """    This method implements the cosine tanimoto coefficient metric    
    :param p_vec: vector one    
    :param q_vec: vector two    
    :return: the tanimoto coefficient between vector one and two    """    
    pq = np.dot(p_vec, q_vec)    
    p_square = np.dot(p_vec,p_vec)#np.linalg.norm(p_vec)    
    q_square = np.dot(q_vec,q_vec)    
    return pq / (p_square + q_square - pq)

def link_prediction(embeddings, walk_length, num_walks, p, q):
    pairs = read_pairs_toPred('data/author_pairs_to_pred_with_index.csv')
    nodes = embeddings.keys()
    score = []
    n = 0
    for pair in pairs:
        if pair[0] in nodes and pair[1] in nodes:
            x = np.array(embeddings[pair[0]])
            y = np.array(embeddings[pair[1]])
            score.append(cosine_similarity(x, y))
        else:
            n += 1 
            score.append(random.random())
    print(n)
    store_result('result_{}_{}_{}_{}_lp.csv'.format(walk_length, num_walks, p, q), score)