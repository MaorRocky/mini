# %%
from pathlib import Path
import community
import numpy as np
import statistics
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections
import matplotlib.ticker as ticker

import time


# function to sort dics by values
def sort_dic(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}


def sort_dic_rev(dic):
    return {k: v for k, v in sorted(dic.items(), reverse=True, key=lambda item: item[1])}


# raeding data
file_to_machines_dic = {}
clean_dict = {}
unknown_set = set()
data_name = 'Obf_oneInTenWeek1_d'
suffix = '.tsv'
G = nx.Graph()

# %%
for i in range(1, 6):
    print('Running data number - {}'.format(i))
    data = pd.read_csv(Path().joinpath('data', data_name + str(i) + suffix), sep='\t',
                       error_bad_lines=False, index_col=False, dtype='unicode')
    data = data.sort_values(by=data.columns[0])

    print('num of rows in data', len(data))

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
        unknown_set.add(file_sha1)
        file_domain = row[domain]
        G.add_edge(file_sha1, file_domain, weight=fileAndDomain_to_machines_dic[(file_sha1, file_domain)])
    print('It took {} seconds.'.format(time.time() - start))

    threat = data.columns[20]
    sha1 = data.columns[3]
    machine = data.columns[13]

    for index, row in data.iterrows():
        file_sha1 = row[sha1]
        machine_guid = row[machine]
        file_threat = row[threat]
        if isinstance(file_threat, str):
            file_to_machines_dic[file_sha1] = file_to_machines_dic.get(file_sha1, []) + [machine_guid]
        else:
            clean_dict[file_sha1] = clean_dict.get(file_sha1, []) + [machine_guid]
# %%
for key, val in file_to_machines_dic.items():
    file_to_machines_dic[key] = len(list(set(val)))

for key, val in clean_dict.items():
    clean_dict[key] = len(list(set(val)))

malicious_dict = {k: v for k, v in file_to_machines_dic.items() if v > 5}
clean_dict = {k: v for k, v in clean_dict.items() if v > 9}
print('number of malicious files:', len(malicious_dict))
print(len(clean_dict), sep='\n')

# %% cleaning the data to
for key, val in malicious_dict.items():
    if key in clean_dict.keys():
        del clean_dict[key]

for file_sha1 in unknown_set.copy():
    if file_sha1 in clean_dict or file_sha1 in malicious_dict:
        unknown_set.remove(file_sha1)

# %%
print("Num of nodes in G {}".format(len(G)))
print('Number of edges in G %s' % (G.number_of_edges()))
lst = list(G.degree)
avg_degree = 0
max_degree = 0
for (item, deg) in lst:
    if deg > max_degree:
        max_degree = deg
    avg_degree += deg
print('avg degree:', round(avg_degree / len(G), 2))
print('max deg:', max_degree)

# print('Average degree G %s' %(np.mean(nx.degree_histogram(G))))
# now we have a graph G which has a edges between files and the domain it was downloaded from, with weight
# which is the number of unique machines which downloaded the file from this domain.

# this is just a print out of the weight of each edge.
attr = nx.get_edge_attributes(G, 'weight')
attr = sort_dic(attr)
# for key, value in attr.items():
#     print(key, ' : ', value)
weight_array = np.array([attr[k] for k in attr])
print('average weight: ', weight_array.mean())
print('max weight :', np.amax(weight_array))
# print("len is ", len(attr))

# %%
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
fig, ax = plt.subplots(figsize=(8, 8))
plt.bar(deg, cnt, color='orange', linewidth=0.6)
plt.yscale('log')
plt.xscale('log')
plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
plt.grid()
# ticks = np.arange(0, 6200, 400)
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticks)
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
print('total communities :', list(partition.values())[-1])
print('average community size:', len(G) / list(partition.values())[-1])
max_community_size_dict = {}
for community_index in partition.values():
    max_community_size_dict[community_index] = max_community_size_dict.get(community_index, 0) + 1
max_community_size_dict = sort_dic_rev(max_community_size_dict)
print('Max community size:', list(max_community_size_dict.values())[0])

# %%
threat_column = data[data.columns[20]]
dirty_precent_per_cluster_lst = []
for file_list in files_per_cluster.values():
    file_list_len = len(file_list)
    counter = 0
    for file in file_list:
        if file in malicious_dict.keys():
            counter += 1
    dirty_precent_per_cluster_lst.append(int(round((counter / file_list_len), 2) * 100))
print(dirty_precent_per_cluster_lst)
# %%
machines_per_cluster = {}
for index, (files_list, domains_list) in enumerate(zip(files_per_cluster.values(), domain_per_cluster.values())):
    for file in files_list:
        for domain in domains_list:
            if G.has_edge(file, domain):
                machines_per_cluster[index] = machines_per_cluster.get(index, 0) + G[file][domain]['weight']

machines_per_cluster = sort_dic(machines_per_cluster)

# print(*machines_per_cluster.items(), sep='\n')

# %%
domain_to_dirty_precent = {}
for index, (files_list, domains_list) in enumerate(zip(files_per_cluster.values(), domain_per_cluster.values())):
    for domain in domains_list:
        domain_total_files_counter = 0
        domain_dirty_files_counter = 0
        for file in files_list:
            if G.has_edge(domain, file):
                domain_total_files_counter += 1
                if file in malicious_dict.keys():
                    domain_dirty_files_counter += 1
        print('%s / %s' % (domain_dirty_files_counter, domain_total_files_counter))
        domain_to_dirty_precent[domain] = int(round((domain_dirty_files_counter / domain_total_files_counter), 2) * 100)

# print(*domain_to_dirty_precent.items(), sep='\n')
# %%
dirty_precent_domains = {}
for domain, percent in domain_to_dirty_precent.items():
    dirty_precent_domains[percent] = dirty_precent_domains.get(percent, 0) + 1

# %%
percent, counter = zip(*dirty_precent_domains.items())  # creating 2 arrays of keys , values
fig, ax = plt.subplots(figsize=(8, 8))
plt.bar(percent, counter, color='blue')
plt.yscale('log')
plt.title("Amount of domains with the number of dirty files percentage")
plt.ylabel("Amount of domains")
plt.xlabel("dirty file percentage")
ticks = np.arange(0, 105, 5)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
plt.savefig('graph_dirty_percent_domains.png')
plt.show()

# %%
# the amount of clusters with different amount of machines
values_to_machines = {}
for key, val in machines_per_cluster.items():
    values_to_machines[val] = values_to_machines.get(val, 0) + 1
print(*values_to_machines.items(), sep='\n')
fig, ax = plt.subplots(figsize=(9, 9))
# values_to_machines = sort_dic(values_to_machines)
t = np.arange(0., len(values_to_machines), 1)
y = [val for val in values_to_machines.values()]
plt.plot(t, y, 'r')
plt.xlabel('Values')
plt.yscale('symlog')
plt.ylabel('Amount of clusters for value X')
ticks = np.arange(1, 223, 20)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
plt.savefig('values_to_machines')
plt.show()
# %%
dirty_per_percent_dict = {}
for i in range(101):
    dirty_per_percent_dict[i] = 0
for i in dirty_precent_per_cluster_lst:
    dirty_per_percent_dict[dirty_precent_per_cluster_lst[i]] += 1

percent, counter = zip(*dirty_per_percent_dict.items())  # creating 2 arrays of keys , values
fig, ax = plt.subplots(figsize=(8, 8))
plt.bar(percent, counter, color='green')
plt.yscale('symlog')
plt.title("Clusters dirty percentage Histogram")
plt.ylabel("Amount of clusters with 'x' dirty files percentage")
plt.xlabel("percetage")
ticks = np.arange(0, 105, 5)
ax.set_xticks(ticks)
ax.set_xticklabels(ticks)
plt.savefig('graph_dirty_percent_clusters.png')
plt.show()
