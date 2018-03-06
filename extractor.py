import os
import sys
import ngrams
import numpy
from sklearn.model_selection import StratifiedKFold


def read_apps_from_folder(path, label):
    loaded_apps = 0
    for file in next(os.walk(path))[2]:
        data.append(path + file)
        labels.append(label)
        loaded_apps += 1
        if loaded_apps >= max_apps_per_technique:
            break


def read_set():
    i = 1
    for folder in os.listdir('dataset/malware/'):
        obf_technique_to_index[folder] = i
        index_to_obf_technique[i] = folder
        read_apps_from_folder('dataset/malware/' + folder + '/', i)
        i += 1
    for folder in os.listdir('dataset/trusted/'):
        read_apps_from_folder('dataset/trusted/' + folder + '/', -obf_technique_to_index[folder])


def convert_special_chars(string):
    for symbol in special_chars_to_char:
        string = string.replace(symbol, special_chars_to_char[symbol])
    return string.splitlines()


def extract_features(set):
    set = ngrams.extract_features(set, features, ngrams_len, frequencies, convert_special_chars)
    return numpy.array(set).astype(numpy.float32)


def save():
    type = 'frequencies' if frequencies else 'occurrences'
    numpy.savez_compressed('dataset/extracted/' + type + '/fold' + str(fold), train=train,
                           test=test, train_labels=train_labels, test_labels=test_labels,
                           train_names=train_names, test_names=test_names)


max_apps_per_technique, ngrams_len = int(sys.argv[1]), int(sys.argv[2])

frequencies = True if sys.argv[3] == 'yes' else False

whiteline = " " * 80
special_chars_to_char = {}
char_to_special_chars = {}
obf_technique_to_index = {}
index_to_obf_technique = {}
data = []
labels = []

with open('special_chars_to_char', 'rb') as f:
    for line in f.read().splitlines():
        special_chars_to_char[line.split(' -> ')[0]] = line.split(' -> ')[1]
        char_to_special_chars[line.split(' -> ')[1]] = line.split(' -> ')[0]

read_set()

with open('dataset/extracted/label_technique_mapper', 'w') as f:
    for key in index_to_obf_technique:
        f.write(str(key) + ":" + index_to_obf_technique[key].upper() + '\n')

skf = StratifiedKFold(n_splits=10)  # numero di fold
fold = 1
for train_indexes, test_indexes in skf.split(data, labels):

    print whiteline + '\r',
    print "Fold ", fold
    train, test, train_labels, test_labels, train_names, test_names = [], [], [], [], [], []
    for i in train_indexes:
        train.append(data[i])
        train_labels.append(labels[i])
        train_names.append(data[i].split('/')[-1])
    for i in test_indexes:
        test.append(data[i])
        test_labels.append(labels[i])
        test_names.append(data[i].split('/')[-1])

    final_ngrams = ngrams.extract_ngrams(train, train_labels, ngrams_len, frequencies, convert_special_chars)

    features = ngrams.select_features(final_ngrams, h=5000, k=2000)
    del final_ngrams

    print "Extracting train features..."
    train = extract_features(train)

    print "Extracting test features..."
    test = extract_features(test)

    save()
    fold += 1
