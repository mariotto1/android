import sys
from sklearn import preprocessing, svm
import numpy
import keras as kr


def normalization(train, test):
    print "Normalizing dataset..."
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(train), scaler.transform(test)


def mlp(train, train_labels, test, test_labels):
    train, test = normalization(train, test)

    # per la codifica one-hot le labels devono essere >=0
    label_offset = 0
    if min(train_labels) < 0:
        label_offset = abs(min(train_labels))
        train_labels = [x + label_offset for x in train_labels]

    model = kr.models.Sequential()
    model.add(kr.layers.Dense(units=128, activation='relu', input_shape=(train.shape[1],)))
    model.add(kr.layers.Dense(units=64, activation='relu'))
    model.add(kr.layers.Dense(units=32, activation='relu'))
    model.add(kr.layers.Dense(units=16, activation='relu'))
    model.add(kr.layers.Dense(units=len(set(train_labels)) + 1))
    model.add(kr.layers.Activation('softmax'))

    # sgd = kr.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train, kr.utils.np_utils.to_categorical(train_labels), epochs=3, batch_size=32, shuffle=True)

    confs = model.predict(test, batch_size=128)
    preds = [numpy.argmax(x) - label_offset for x in confs]
    score = calculate_score_techniquewise(preds, test_labels)
    save_fold_results(confs, preds, test_labels, score)
    return score


def SVM(train, train_labels, test, test_labels, cost=1.0, ker='rbf', verb=False):
    train, test = normalization(train, test)
    print "Training svm..."
    clf = svm.SVC(C=cost, kernel=ker, decision_function_shape='ovr', verbose=verb)
    clf.fit(train, train_labels)
    confs = numpy.array(clf.decision_function(test))
    preds = numpy.array(clf.predict(test))
    score = calculate_score_techniquewise(preds, test_labels)
    save_fold_results(confs, preds, test_labels, score)
    return score


def calculate_score(preds, test_labels, label_threshold):
    score = {'corretti': 0, 'FN': 0, 'FP': 0, 'tecnica_sbagliata': 0, 'positivi': 0, 'negativi': 0}
    for pred, label in zip(preds, test_labels):
        if pred == label:
            score['corretti'] += 1
        elif (pred >= label_threshold and label >= label_threshold) or (pred < label_threshold and label < label_threshold):
            score['tecnica_sbagliata'] += 1
        elif pred >= label_threshold and label < label_threshold:
            score['FP'] += 1
        elif pred < label_threshold and label >= label_threshold:
            score['FN'] += 1
    score['positivi'] = len(preds) / 2
    score['negativi'] = len(preds) / 2
    score['FPR'] = score['FP'] / float(score['negativi'])
    score['FNR'] = score['FN'] / float(score['positivi'])
    return score


def calculate_score_techniquewise(preds, test_labels):
    score = {}
    for technique in technique_label_mapper:
        score[technique] = sum(1 for x in test_labels if abs(x) == technique_label_mapper[technique])
        score[technique + " corretti"] = sum(1 for i, j in zip(preds, test_labels) if technique_label_mapper[technique] == abs(j) and abs(i) == abs(j))
        score[technique + ' accuratezza'] = score[technique + " corretti"] / float(score[technique])
    score['totali'] = len(test_labels)
    score['corretti'] = sum(score[technique + " corretti"] for technique in technique_label_mapper)
    score['accuratezza pesata'] = score['corretti'] / float(score['totali'])
    return score


def save_fold_results(confs, preds, test_labels, score):
    path = "results/" + classifier + "/" + type + "/"
    with open(path + 'confidences', 'a') as f:
        f.write("FOLD " + str(fold) + '\n')
        f.write(';'.join(str(x).replace('\n', '') for x in confs) + '\n')
        f.write(';'.join(str(x) for x in test_labels) + '\n')
    with open(path + 'predictions', 'a') as f:
        f.write("FOLD " + str(fold) + '\n')
        f.write(' '.join(str(x) for x in preds) + '\n')
        f.write(' '.join(str(x) for x in test_labels) + '\n')
    with open(path + 'scores', 'a') as f:
        f.write("FOLD " + str(fold) + '\n')
        f.write('\n'.join(x + ': ' + str(score[x]) for x in sorted(score)) + '\n')


classifier = sys.argv[1]
frequencies = True if sys.argv[2] == 'yes' else False
type = 'frequencies' if frequencies else 'occurrences'
global_score = {}
label_technique_mapper = {}
technique_label_mapper = {}

with open('dataset/extracted/label_technique_mapper', 'r') as f:
    for line in f.read().splitlines():
        label_technique_mapper[int(line.split(':')[0])] = line.split(':')[1].upper()
        technique_label_mapper[line.split(':')[1].upper()] = int(line.split(':')[0])

for fold in range(1, 11):
    print "Fold", fold
    load = numpy.load('dataset/extracted/' + type + '/fold' + str(fold) + '.npz')

    # considero solo la tecnica di offuscamento, non malware/trusted
    train_labels = [abs(x) for x in load['train_labels']]
    test_labels = [abs(x) for x in load['test_labels']]
    train = load['train']
    test = load['test']

    if classifier == 'svm':
        fold_score = SVM(train, train_labels, test, test_labels)
    elif classifier == 'mlp':
        fold_score = mlp(train, train_labels, test, test_labels)

    for key in fold_score:
        if 'accuratezza' not in key:
            if key not in global_score:
                global_score[key] = fold_score[key]
            else:
                global_score[key] += fold_score[key]

for technique in technique_label_mapper:
    global_score[technique + ' accuratezza'] = global_score[technique + " corretti"] / float(global_score[technique])
global_score['accuratezza pesata'] = global_score['corretti'] / float(global_score['totali'])

with open('results/' + classifier + '/final_results_' + type, 'w') as f:
    f.write('\n'.join(x + ': ' + str(global_score[x]) for x in sorted(global_score)) + '\n')
