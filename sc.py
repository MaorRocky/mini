import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import regex as re


pattern = re.compile(r'([\p{IsHan}\p{IsBopo}\p{IsHira}\p{IsKatakana}]+)', re.UNICODE)


class File:
    def __init__(self, index, filename, machine, threat, domain, num_of_machines):
        self.index = index
        self.filename = filename
        self.machine = machine
        self.threat = threat
        self.domain = domain
        self.num_of_machines = num_of_machines

    def __str__(self):
        return "index is {}, name is {}, machine is {},threat is {}, domain is {}," \
               "num_of_machines is {}".format(self.index,
                                              self.filename,
                                              self.machine,
                                              self.threat,
                                              self.domain,
                                              self.num_of_machines)


def isNaN(num):
    return num != num


data = pd.read_csv('/home/maor/PycharmProjects/mini/Obf_oneInTenWeek1_d1.tsv', sep='\t',
                   error_bad_lines=False, index_col=False, dtype='unicode')

columns = data.columns
i = list(columns.values).index('1/1/2017 7:18:01 PM')  # i will return index of 2, which is 1

threats_column = data[data.columns[20]]
machine_column = data[data.columns[13]]
file_names_column = data[data.columns[0]]
domains_column = data[data.columns[17]]
infected_index = []
virus_name = set()

for i, f in enumerate(threats_column):
    if not isNaN(f):
        virus_name.add(f)
        infected_index.append(i)

infected_names_set = set()
# for i in infected_index:
#     infected_names_set.add(file_names_column[i])
print("number of names {} ".format(len(infected_names_set)))
print("len of infected_index {}".format(len(infected_index)))
files = []
for index in infected_index:
    row = data.loc[[index]]
    name = row[data.columns[0]].iloc[0]
    machine = row[data.columns[13]].iloc[0]
    if (name, machine) in infected_names_set:
        continue
    else:
        infected_names_set.add((name, machine))
        threat = row[data.columns[20]].iloc[0]
        domain = row[data.columns[17]].iloc[0]
        file = File(index, name, machine, threat, domain, 0)
        files.append(file)

print("files length is ", len(files))

fileName_to_machines_dict = {}
for file in files:
    if file.filename in fileName_to_machines_dict:
        fileName_to_machines_dict[file.filename] += 1
    else:
        fileName_to_machines_dict[file.filename] = 1
print(len(fileName_to_machines_dict))
fileName_to_machines_dict = {key: val for key, val in fileName_to_machines_dict.items() if val > 20}

# for key, val in fileName_to_machines_dict.items():
#     new_key = pattern.sub('x', key)
#     fileName_to_machines_dict[new_key] = fileName_to_machines_dict.pop(key)

fileName_to_machines_dict = {k: v for k, v in sorted(fileName_to_machines_dict.items(), key=lambda item: item[1])}

print("myDict:\n", fileName_to_machines_dict)
print("size of fileName_to_machines_dict: ", len(fileName_to_machines_dict))

files_after_filter = []
file_names_set = set()
for file in files:
    if file.filename in fileName_to_machines_dict:
        file.num_of_machines = fileName_to_machines_dict.get(file.filename)
        if file.filename not in file_names_set:
            file_names_set.add(file.filename)
            files_after_filter.append(file)
plt.rcParams["font.family"] = "STSong"
G = nx.Graph()
for file in files_after_filter:
    G.add_edge(file.filename, file.domain)

print(nx.degree(G)) 

plt.subplot(121)
nx.draw_random(G, with_labels=True, font_weight='bold', font_family='STSong', font_size=10, node_size=100)
plt.savefig('graph.png')
plt.show()
