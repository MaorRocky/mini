# %%
from pathlib import Path
import community

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 15))
data_name = 'Obf_oneInTenWeek1_d'
suffix = '.tsv'
G = nx.Graph()
# for i in range(1, 2):
# print('Running data number - {}'.format(i))
data = pd.read_csv(Path().joinpath('data', data_name + str(1) + suffix), sep='\t',
                   error_bad_lines=False, index_col=False, dtype='unicode')

print(len(data))

name = data.columns[0]
domain = data.columns[17]
machine = data.columns[13]
data = data.sort_values(by=data.columns[0])
machine_dic = {}
for index, row in data.iterrows():
    fileName = row[name]
    file_domain = row[domain]
    machine_dic[(fileName, file_domain)] = machine_dic.get((fileName, file_domain), 0) + 1
# machine_dic = {k: v for k, v in sorted(machine_dic.items(), key=lambda item: item[1])}
# print(machine_dic)


for index, row in data.iterrows():
    fileName = row[name]
    file_domain = row[domain]
    G.add_edge(fileName, file_domain, weight=machine_dic[(fileName, file_domain)])

# %%
attr = nx.get_edge_attributes(G, 'weight')
# attr = {key: value for (key, value) in attr.items() if value > 100}
attr = {k: v for k, v in sorted(attr.items(), key=lambda item: item[1])}
for key, value in attr.items():
    print(key, value)
print("len is ", len(attr))

# %%
lst = {}
print("Num of nodes in G {}".format(len(G)))
# for tup in nx.degree(G):
#     lst[tup[0]] = tup[1]
# lst = {k: v for k, v in sorted(lst.items(), key=lambda item: item[1])}
# print(lst)
# Find modularity
partition = community.best_partition(G)
partition = {k: v for k, v in sorted(partition.items(), key=lambda item: item[1])}

file_names_column = data[data.columns[0]]
fileName_set = set(file_names_column)
# %%
for key, val in partition.items():
    print(key, ':', val)
# %%
domain_per_cluster = {}
files_per_cluster = {}
for key, val in partition.items():
    if key in fileName_set:
        files_per_cluster[val] = files_per_cluster.get(val, []) + [key]
    else:
        domain_per_cluster[val] = domain_per_cluster.get(val, []) + [key]

# files_per_cluster = {k: v for k, v in sorted(files_per_cluster.items(), key=lambda item: item[1])}

for key, val in domain_per_cluster.items():
    print(key, ':', val)

# %%
threat = data.columns[20]
name = data.columns[0]

threat_files_set = set()
for index, row in data.iterrows():
    file_threat = row[threat]
    if isinstance(file_threat, str):
        threat_files_set.add(row[name])

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
index = 0
for files_list, domains_list in zip(files_per_cluster.values(), domain_per_cluster.values()):
    for file in files_list:
        for domain in domains_list:
            if G.has_edge(file, domain):
                machines_per_cluster[index] = machines_per_cluster.get(index, 0) + G[file][domain]['weight']
    index += 1

machines_per_cluster = {k: v for k, v in sorted(machines_per_cluster.items(), key=lambda item: item[1])}

for key, value in machines_per_cluster.items():
    print(key, value)
# %%
