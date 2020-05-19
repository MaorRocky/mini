# %%
from pathlib import Path
import community
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
import matplotlib.ticker as ticker

import time


# function to sort dics by values
def sort_dic(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}


# raeding data
data_name = 'Obf_oneInTenWeek1_d'
suffix = '.tsv'
G = nx.Graph()
# for i in range(1, 2):
# print('Running data number - {}'.format(i))
data = pd.read_csv(Path().joinpath('data', data_name + str(1) + suffix), sep='\t',
                   error_bad_lines=False, index_col=False, dtype='unicode')
data = data.sort_values(by=data.columns[0])

print('num of rows in data', len(data))
# %%
# instead of using names we will use sha1
# Number of distinct machines file was downloaded to from this domain. this will be the weight of and edge
# name = data.columns[0]
start = time.time()  # just to know how much time it runs.
# fileAndDomain_to_machines_dic key:val -> (key) file&domain : (val) num of machines
sha1 = data.columns[3]
domain = data.columns[17]
machine = data.columns[13]
fileAndDomain_to_machines_dic = {}
for index, row in data.iterrows():
    file_sha1 = row[sha1]
    file_domain = row[domain]
    machine_guid = row[machine]
    fileAndDomain_to_machines_dic[(file_sha1, file_domain)] = fileAndDomain_to_machines_dic.get(
        (file_sha1, file_domain), []) + [machine_guid]

for key, val in fileAndDomain_to_machines_dic.items():
    fileAndDomain_to_machines_dic[key] = len(list(set(val)))
fileAndDomain_to_machines_dic = sort_dic(fileAndDomain_to_machines_dic)

for index, row in data.iterrows():
    file_sha1 = row[sha1]
    file_domain = row[domain]
    G.add_edge(file_sha1, file_domain, weight=fileAndDomain_to_machines_dic[(file_sha1, file_domain)])
print('It took {} seconds.'.format(time.time() - start))
print("Num of nodes in G {}".format(len(G)))
# now we have a graph G which has a edges between files and the domain it was downloaded from, with weight
# which is the number of unique machines which downloaded the file from this domain.
# %%
# this is just a print out of the weight of each edge.
attr = nx.get_edge_attributes(G, 'weight')
attr = sort_dic(attr)
for key, value in attr.items():
    print(key, ' : ', value)
print("len is ", len(attr))
# %%
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = plt.subplots(figsize=(10, 10))
plt.scatter(deg, cnt, color='b', linewidths=0.1)
plt.yscale('symlog')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ticks = np.arange(0,6200,300)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
plt.savefig('graph_degree_histogram.png')
plt.show()

# %%
partition = community.best_partition(G, weight='weight')
partition = sort_dic(partition)

# file_names_column = data[data.columns[0]]
# fileName_set = set(file_names_column)

file_sah1_column = data[data.columns[3]]
file_sah1_set = set(file_sah1_column)
domain_per_cluster = {}
files_per_cluster = {}
for key, val in partition.items():
    if key in file_sah1_set:
        files_per_cluster[val] = files_per_cluster.get(val, []) + [key]
    else:
        domain_per_cluster[val] = domain_per_cluster.get(val, []) + [key]

# %%
threat = data.columns[20]
# name = data.columns[0]
sha1 = data.columns[3]

threat_files_set = set()
for index, row in data.iterrows():
    file_threat = row[threat]
    if isinstance(file_threat, str):
        threat_files_set.add(row[sha1])

threat_column = data[data.columns[20]]
dirty_files_lst = []
for file_list in files_per_cluster.values():
    file_list_len = len(file_list)
    counter = 0
    for file in file_list:
        if file in threat_files_set:
            counter += 1
    dirty_files_lst.append(counter / file_list_len)
print(dirty_files_lst)
# %%
machines_per_cluster = {}
for index, (files_list, domains_list) in enumerate(zip(files_per_cluster.values(), domain_per_cluster.values())):
    for file in files_list:
        for domain in domains_list:
            if G.has_edge(file, domain):
                machines_per_cluster[index] = machines_per_cluster.get(index, 0) + G[file][domain]['weight']

machines_per_cluster = sort_dic(machines_per_cluster)

print(*machines_per_cluster.items(), sep='\n')
