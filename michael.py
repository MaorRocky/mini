from pathlib import Path
import community
import numpy as np
import statistics
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
from sklearn import tree
from numpy import nan
import matplotlib.ticker as ticker

import time

G = nx.Graph()


def add_edge(u, v, w):
    if G.has_edge(u, v):
        w = G[u][v]['weight'] + w
        G.add_edge(u, v, weight=w)
    else:
        G.add_edge(u, v, weight=w)


add_edge(1, 2, 3)
add_edge(1, 2, 10)
add_edge(1, 3, 10)
add_edge(1, 3, 10)
edge_to_weights_dic = nx.get_edge_attributes(G, 'weight')
