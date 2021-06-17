import csv
import numpy as np

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

def cosine_similarity1(x, y):
    num = x.dot(y.T)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return num / denom

def cosine_similarity2(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0

def cosine_similarity3(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def tanimoto_coefficient(p_vec, q_vec):    
    """    This method implements the cosine tanimoto coefficient metric    
    :param p_vec: vector one    
    :param q_vec: vector two    
    :return: the tanimoto coefficient between vector one and two    """    
    pq = np.dot(p_vec, q_vec)    
    p_square = np.dot(p_vec,p_vec)#np.linalg.norm(p_vec)    
    q_square = np.dot(q_vec,q_vec)    
    return pq / (p_square + q_square - pq)

def link_prediction(author_embeddings, author_ids, walk_length, n, vec_size):
    pairs = read_pairs_toPred('data/author_pairs_to_pred_with_index.csv')
    score = []
    for pair in pairs:
        i = author_ids.index(pair[0])
        j = author_ids.index(pair[1])
        x = np.array(author_embeddings[i])
        y = np.array(author_embeddings[j])
        score.append(cosine_similarity1(x, y))
    store_result('result_meta_lp_{}_{}_{}.csv'.format(walk_length, n, vec_size), score)