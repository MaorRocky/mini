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


def load_data_and_creat_graph(G, file_to_machines_dic, clean_dict, unknown_set):
    data_name = 'Obf_oneInTenWeek1_d'
    suffix = '.tsv'
    for i in range(1, 6):
        print('Running data number - {}'.format(i))
        data = pd.read_csv(Path().joinpath('data', data_name + str(i) + suffix), sep='\t',
                           error_bad_lines=False, index_col=False, dtype='unicode')
        data = data.sort_values(by=data.columns[0])
        # instead of using names we will use sha1
        # Number of distinct machines file was downloaded to from this domain. this will be the weight of and edge
        # name = data.columns[0]
        start = time.time()  # just to know how much time it runs.
        # fileAndDomain_to_machines_dic key:val -> (key) file&domain : (val) num of machines
        sha1 = data.columns[3]
        machine = data.columns[13]
        domain = data.columns[17]
        threat = data.columns[20]

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

        for index, row in data.iterrows():
            file_sha1 = row[sha1]
            machine_guid = row[machine]
            file_threat = row[threat]
            if isinstance(file_threat, str):
                file_to_machines_dic[file_sha1] = file_to_machines_dic.get(file_sha1, []) + [machine_guid]
            else:
                clean_dict[file_sha1] = clean_dict.get(file_sha1, []) + [machine_guid]

        for key, val in file_to_machines_dic.items():
            file_to_machines_dic[key] = len(list(set(val)))

        for key, val in clean_dict.items():
            clean_dict[key] = len(list(set(val)))

        malicious_dict = {k: v for k, v in file_to_machines_dic.items() if v > 5}
        clean_dict = {k: v for k, v in clean_dict.items() if v > 9}
        print('number of malicious files:', len(malicious_dict))
        print(len(clean_dict), sep='\n')

        return G, fileAndDomain_to_machines_dic, malicious_dict, clean_dict


if __name__ == '__main__':
    print('start running')
    G = nx.Graph()
    file_to_machines_dic = {}
    clean_dict = {}
    unknown_set = set()
    G, fileAndDomain_to_machines_dic, malicious_dict, clean_dict = load_data_and_creat_graph(G, file_to_machines_dic,
                                                                                             clean_dict, unknown_set)
