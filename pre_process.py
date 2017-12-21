import re
import pickle


def descriptor(entry):
    l = entry.replace('\n', '')
    w123 = re.split('\[', l)
    if len(w123) < 2:
        print( entry)
        return None

    w12 = re.split('\s|\\t|\r', w123[0])
    w12 = list(filter(None, w12))

    w3 = re.split('\s|\\t|\r', w123[1])
    w3 = list(filter(None, w3))
    w3 = [float(x) for x in w3]
    id = int(w12[0])
    word = w12[1]
    if id % 502 == 0:
        print(int(id / 502))
    return id, word, w3


data = open("pt.tsv", encoding="utf8").read()

lns = data.split("]")

all_data = [descriptor(l) for l in lns]
all_data = list(filter(None, all_data))
pickle.dump(all_data, open("wordvector.p", "wb"))
