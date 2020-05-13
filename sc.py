# %%
from pathlib import Path
import community
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
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

print('num of rows in data', len(data))
# %%
# instead of using names we will use sha1
# Number of distinct machines file was downloaded to from this domain. this will be the weight of and edge
# name = data.columns[0]
start = time.time()
sha1 = data.columns[3]
domain = data.columns[17]
machine = data.columns[13]
data = data.sort_values(by=data.columns[0])
machines_dic = {}
for index, row in data.iterrows():
    file_sha1 = row[sha1]
    file_domain = row[domain]
    machine_guid = row[machine]
    machines_dic[(file_sha1, file_domain)] = machines_dic.get((file_sha1, file_domain), []) + [machine_guid]

for key, val in machines_dic.items():
    machines_dic[key] = len(list(set(val)))
machines_dic = sort_dic(machines_dic)

for index, row in data.iterrows():
    file_sha1 = row[sha1]
    file_domain = row[domain]
    G.add_edge(file_sha1, file_domain, weight=machines_dic[(file_sha1, file_domain)])
print('It took {} seconds.'.format(time.time() - start))
print("Num of nodes in G {}".format(len(G)))
# %%
attr = nx.get_edge_attributes(G, 'weight')
attr = sort_dic(attr)
for key, value in attr.items():
    print(key, ' : ', value)
print("len is ", len(attr))
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
# %%
plt.figure(figsize=(30, 20))
t = np.arange(0., len(dirty_files_lst), 1)

# plt.plot(t, dirty_files_lst, 'r--', )
plt.scatter(np.arange(0.0, len(dirty_files_lst), 1.0), dirty_files_lst)
plt.xlabel('cluster number')
plt.savefig('graph.png')
plt.show()
# %%

t = np.arange(0., len(files_per_cluster), 1)
plt.figure(figsize=(10, 5))

files_per_cluster_lst = []
for key, value in files_per_cluster.items():
    files_per_cluster_lst.append(len(value))
# plt.plot(t, dirty_files_lst, 'r--', )
plt.scatter(np.arange(0.0, len(files_per_cluster), 1.0), files_per_cluster_lst, linewidth=1)
plt.xlabel('cluster number')
plt.ylabel('files per cluster')
plt.xscale('log')
plt.savefig('graph.png')
plt.show()
