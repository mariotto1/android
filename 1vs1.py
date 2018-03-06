import sys
import multiprocessing as mp
from sklearn import preprocessing, svm
import numpy
import keras as kr


def normalization(train, test):
    print "Normalizing dataset..."
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def mlp(train, train_labels, test, test_labels):
    train, test = normalization(train, test)

    model = kr.models.Sequential()
    model.add(kr.layers.Dense(units=128, activation='relu', input_shape=(train.shape[1],)))
    model.add(kr.layers.Dense(units=64, activation='relu'))
    model.add(kr.layers.Dense(units=32, activation='relu'))
    model.add(kr.layers.Dense(units=16, activation='relu'))
    model.add(kr.layers.Dense(units=len(set(train_labels))))
    model.add(kr.layers.Activation('softmax'))

    # sgd = kr.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train, kr.utils.np_utils.to_categorical(train_labels), epochs=3, batch_size=32, shuffle=True)

    confs = model.predict(test, batch_size=128)
    preds = [numpy.argmax(x) for x in confs]
    score = calculate_score(preds, test_labels)
    save_fold_results(confs, preds, test_labels, score)
    return score


def SVM(train, train_labels, test, test_labels, return_list, lock, cost=1.0, ker='rbf', verb=False):
    train, test = normalization(train, test)
    print "Training svm..."
    clf = svm.SVC(C=cost, kernel=ker, verbose=verb)
    clf.fit(train, train_labels)
    confs = numpy.array(clf.decision_function(test))
    preds = numpy.array(clf.predict(test))
    score = calculate_score(preds, test_labels)
    lock.acquire()
    save_fold_results(confs, preds, test_labels, score)
    lock.release()
    return_list.append(score)


def calculate_score(preds, test_labels):
    score = {'Corretti': 0, 'FN': 0, 'FP': 0}
    for pred, label in zip(preds, test_labels):
        if pred == label:
            score['Corretti'] += 1
        elif pred > label:
            score['FP'] += 1
        elif pred < label:
            score['FN'] += 1
    score['Positivi'] = sum(test_labels)
    score['Negativi'] = len(test_labels) - score['Positivi']
    score['FPR'] = score['FP'] / float(score['Negativi'])
    score['FNR'] = score['FN'] / float(score['Positivi'])
    score['Accuratezza'] = score['Corretti'] / float(score['Positivi'] + score['Negativi'])
    return score


def save_fold_results(confs, preds, test_labels, score):
    path = '/'.join(["results_1vs1", classifier, technique, type])
    with open(path + '/confidences', 'a') as f:
        f.write("FOLD " + str(fold) + '\n')
        f.write(';'.join(str(x).replace('\n', '') for x in confs) + '\n')
        f.write(';'.join(str(x) for x in test_labels) + '\n')
    with open(path + '/predictions', 'a') as f:
        f.write("FOLD " + str(fold) + '\n')
        f.write(' '.join(str(x) for x in preds) + '\n')
        f.write(' '.join(str(x) for x in test_labels) + '\n')
    with open(path + '/scores', 'a') as f:
        f.write("FOLD " + str(fold) + '\n')
        f.write('\n'.join(x + ': ' + str(score[x]) for x in sorted(score)) + '\n')


def to_binary_labels(labels_list):
    labels = []
    if technique == 'RAW':
        for label in labels_list:
            labels.append(1 if label_technique_mapper[abs(label)] == technique else 0)
    else:
        for label in labels_list:
            labels.append(1 if label_technique_mapper[abs(label)] == technique or label_technique_mapper[abs(label)] == 'ALL' else 0)
    return labels


classifier = sys.argv[1]
frequencies = True if sys.argv[2] == 'yes' else False
type = 'frequencies' if frequencies else 'occurrences'
label_technique_mapper = {}
technique_label_mapper = {}

with open('dataset/extracted/label_technique_mapper', 'r') as f:
    for line in f.read().splitlines():
        label_technique_mapper[int(line.split(':')[0])] = line.split(':')[1]
        technique_label_mapper[line.split(':')[1]] = int(line.split(':')[0])

for technique in technique_label_mapper:
    if technique != 'ALL':
        global_score = {'Corretti': 0, 'FN': 0, 'FP': 0, 'Negativi': 0, 'Positivi': 0}
        manager = mp.Manager()
        return_list = manager.list()
        lock = mp.Lock()
        jobs = []
        for fold in range(1, 11):
            print technique, ", Fold", fold
            load = numpy.load('dataset/extracted/' + type + '/fold' + str(fold) + '.npz')

            train = load['train']
            test = load['test']
            # train_labels = to_binary_labels(load['train_labels'])
            test_labels = to_binary_labels(load['test_labels'])

            # toglie ALL dal training set
            train_labels = load['train_labels']
            train = numpy.delete(train, [x for x in range(len(train_labels)) if label_technique_mapper[abs(train_labels[x])] == 'ALL'], 0)
            train_labels = [x for x in train_labels if abs(x) != technique_label_mapper['ALL']]
            train_labels = to_binary_labels(train_labels)

            if classifier == 'svm':
                p = mp.Process(target=SVM, args=(train, train_labels, test, test_labels, return_list, lock))
                jobs.append(p)
                p.start()
                # fold_score = SVM(train, train_labels, test, test_labels)
            elif classifier == 'mlp':
                return_list.append(mlp(train, train_labels, test, test_labels))

        if len(jobs) > 0:
            for proc in jobs:
                proc.join()
        for score in return_list:
            for key in global_score:
                global_score[key] += score[key]

        global_score['FPR'] = global_score['FP'] / float(global_score['Negativi'])
        global_score['FNR'] = global_score['FN'] / float(global_score['Positivi'])
        global_score['Accuratezza'] = global_score['Corretti'] / float(global_score['Positivi'] + global_score['Negativi'])

        with open('/'.join(['results_1vs1', classifier, technique, 'final_results_' + type]), 'w') as f:
            f.write('\n'.join(x + ': ' + str(global_score[x]) for x in sorted(global_score)) + '\n')

        with open('/'.join(['results_1vs1', classifier, 'global_results_' + type]), 'a') as f:
            f.write(technique + '\n')
            f.write('\n'.join(x + ': ' + str(global_score[x]) for x in sorted(global_score)) + '\n\n')
