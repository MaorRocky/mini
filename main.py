# %%
from pathlib import Path
import community

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def isNaN(num):
    return num != num


plt.figure(figsize=(20, 15))
data_name = 'Obf_oneInTenWeek1_d'
suffix = '.tsv'
G = nx.Graph()
for i in range(1, 3):
    print('Running data number - {}'.format(i))
    data = pd.read_csv(Path().joinpath('data', data_name + str(i) + suffix), sep='\t',
                       error_bad_lines=False, index_col=False, dtype='unicode')
    domains_column = data[data.columns[17]]
    file_names_column = data[data.columns[0]]

    num_of_max_files = 20
    fileName_dic = {}
    row_index = []
    for index, row in data.iterrows():
        if not isNaN(row[20]):
            fileName = row[0]
            fileName_dic[fileName] = fileName_dic.get(fileName, 0) + 1
            row_index.append(index)
    fileName_dic = {k: v for k, v in sorted(fileName_dic.items(), key=lambda item: item[1])}
    print("number of files which poses a threat: {}".format(len(fileName_dic)))
    fileName_dic = dict(list(fileName_dic.items())[-num_of_max_files:])

    fileName_2_domain = {}
    for index in row_index:
        row = data.iloc[index]
        name = row[0]
        domain = row[17]
        if name in fileName_dic:
            fileName_2_domain[(name, domain)] = fileName_2_domain.get((name, domain), 0) + 1
    print("number of domains from the {} most downloaded files: {}".format(num_of_max_files, len(fileName_2_domain)))

    for (name, domain) in fileName_2_domain:
        G.add_edge(name, domain)
    # nx.draw_random(G, with_labels=True, font_weight='bold', font_family='STSong', font_size=16, node_size=40)

# %%

partition = community.best_partition(G)

size = float(len(set(partition.values())))
pos = nx.spring_layout(G)
count = 0.
for com in set(partition.values()):
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                  if partition[nodes] == com]
    nx.draw_networkx_nodes(G, pos, list_nodes,with_labels=True, font_weight='bold',font_size=14, font_family='STSong', node_size=40,
                           node_color=str(count / size))

nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
