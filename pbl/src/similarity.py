# src/similarity.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

def build_feature_matrix(df):
    # df: aggregated by area (State, District, Year aggregated into totals)
    # return matrix with rows=area, cols=crime totals
    agg = df.groupby(['STATE/UT', 'DISTRICT'])[[
        "Rape","Kidnapping and Abduction","Dowry Deaths",
        "Assault on women with intent to outrage her modesty",
        "Insult to modesty of Women","Cruelty by Husband or his Relatives"
    ]].sum()
    return agg

def recommend_similar(area_row, matrix, topn=5):
    # area_row: index tuple (state, district)
    X = matrix.values
    sim = cosine_similarity(X)
    idx = list(matrix.index).index(area_row)
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recs = [(matrix.index[i], s) for i, s in scores[1:topn+1]]
    return recs

def build_similarity_graph(matrix, threshold=0.7):
    X = matrix.values
    sim = cosine_similarity(X)
    G = nx.Graph()
    for idx, name in enumerate(matrix.index):
        G.add_node(name)
    n = len(matrix)
    for i in range(n):
        for j in range(i+1, n):
            if sim[i,j] >= threshold:
                G.add_edge(matrix.index[i], matrix.index[j], weight=float(sim[i,j]))
    return G
