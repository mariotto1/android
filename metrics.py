import sys
import os
import numpy
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d


def read_softmax(confs):
    l = []
    for x in confs.split(';'):
        x = x[1:-1].split(' ')
        x = filter(lambda a: a != '', x)
        l.append([float(y) for y in x])
    return l


def read_confidences(confs):
    return [float(x) for x in confs.split(';')]


def softmax_to_probability(confs):
    return [conf[1] for conf in confs]


def hyperplane_distances_to_probability(confs):
    pass


def plot_roc():
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(technique)
    plt.legend(loc="lower right")
    plt.show()


classifier = sys.argv[1]
frequencies = True if sys.argv[2] == 'yes' else False

results = {}

for results_folder in ['results_1vs1_with_ALL']:  # ,'results_1vs1']:
    print results_folder
    for technique in next(os.walk(results_folder + '/' + classifier))[1]:
        with open('/'.join([results_folder, classifier, technique, 'frequencies' if frequencies else 'occurrences', 'confidences']), 'r') as f:
            lines = f.read().splitlines()
        confidences = []
        labels = []
        for i in range(0, len(lines), 3):
            if classifier == 'mlp':
                confidences.extend(softmax_to_probability(read_softmax(lines[i + 1])))
            elif classifier == 'svm':
                confidences.extend(read_confidences(lines[i + 1]))
            labels.extend([int(x) for x in lines[i + 2].split(';')])
        assert len(labels) == len(confidences)
        roc_auc = roc_auc_score(labels, confidences)
        fpr, tpr, thresholds = roc_curve(labels, confidences)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # thresh = interp1d(fpr, thresholds)(eer)
        if technique not in results:
            results[technique] = [roc_auc, eer]
        else:
            results[technique] = results[technique] + [roc_auc, eer]

with open('metrics_' + classifier + '.csv', 'w') as f:
    f.write(",w/ ALL,,w/o ALL,\n")
    f.write(",AUC,EER,AUC,EER\n")
    for technique in results:
        f.write(technique + ',' + ','.join(str(x) for x in results[technique]) + '\n')
