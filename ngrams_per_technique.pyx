from libcpp cimport bool
from cython.parallel import parallel, prange
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map as cpp_map
from cython.operator cimport dereference as deref, preincrement as inc
import sys
import multiprocessing

cdef int batch_size=1000
cdef int n_core=multiprocessing.cpu_count()

cdef cpp_map[string,float] extract_ngrams_app(vector[string]& app, int ngram_max_len, bool frequencies) nogil:
    cdef cpp_map[string,float] app_ngrams
    cdef string line
    cdef int i
    cdef int window_size
    cdef int app_len=0
    
    if frequencies:
        for line in app:
            app_len+=line.length()

    for line in app:
        for window_size in range(1,ngram_max_len+1):
            if window_size<=line.length():
                for i in range(line.length() - window_size + 1):
                    if not frequencies:
                        app_ngrams[line.substr(i,window_size)]+=1.0
                    else:
                        app_ngrams[line.substr(i,window_size)]+=1.0/app_len
    return app_ngrams


def extract_ngrams(list data, list labels, int ngram_max_len,bool frequencies, list obf_techniques):  
    cdef:
        vector[vector[string]] data_vect
        vector[int] labels_vect
        int data_len
        list total_ngrams=[]
        cpp_map[string,float] ngrams_pos
        cpp_map[string,float] ngrams_neg
        cpp_map[string,float] app_ngrams
        int j
        int k
        string ngram
        float value
        float pos
        float neg
        cpp_map[string,float].iterator it
        cpp_map[int,cpp_map[string,float]] technique_ngrams
        dict technique_labels={}
    
    print "Extracting ngrams..."
    for j in range(0,len(data),batch_size):
        sys.stdout.write('Reading batch from disk...\r')
        sys.stdout.flush()     
        data_vect.clear()
        for path in data[j:j+batch_size]:
            with open(path , 'rb') as f:
                data_vect.push_back(f.read().splitlines())
        labels_vect = labels[j:j+batch_size]
        data_len = data_vect.size()
        sys.stdout.write(' '*80+'\r')
        sys.stdout.flush()

        for tech in obf_techniques:
            technique_labels[tech]=[1 if x==tech else 0 for x in labels_vect]

        with nogil, parallel(num_threads=n_core):
            for k in prange(data_len,schedule='dynamic'):
                app_ngrams=extract_ngrams_app(data_vect[k],ngram_max_len,frequencies)
                it = app_ngrams.begin()
                while it != app_ngrams.end():
                    ngram,value = deref(it).first, deref(it).second
                    with gil:
                        for tech in obf_techniques: 
                            if technique_labels[tech][k]>0:
                                technique_ngrams[tech][ngram]+=value
                                technique_ngrams[-tech][ngram]+=0  #per inizializzarlo se non esiste
                            elif technique_labels[tech][k]<=0:
                                technique_ngrams[tech][ngram]+=0  #per inizializzarlo se non esiste
                                technique_ngrams[-tech][ngram]+=value
                    inc(it)
                with gil:
                    sys.stdout.write('{}/{} apps\r'.format(k+j+1,len(data)))
                    sys.stdout.flush()
    for tech in obf_techniques:
        tech_total_ngrams=[]
        it = technique_ngrams[tech].begin()
        pos_len=labels.count(tech)
        neg_len=len(labels)-pos_len
        while it != technique_ngrams[tech].end():
            ngram, value = deref(it).first, deref(it).second
            pos=value/pos_len
            neg=technique_ngrams[-tech][ngram]/neg_len
            if abs(neg-pos)/max(neg,pos) < 1.0:
                tech_total_ngrams.append((ngram,abs(neg-pos)/max(neg,pos)))
            inc(it)
        total_ngrams.append(tech_total_ngrams)
        tech_total_ngrams=[]
    return total_ngrams


def select_features(list final_ngrams_list,list obf_techniques,int h,int k):
    print "Selecting top {} features...".format(k)
    cdef list selected_features=[]
    cdef int i
    cdef int j
    for final_ngrams in final_ngrams_list:
        final_ngrams.sort(key=lambda tup: tup[1],reverse=True)
        final_ngrams = [x[0] for x in final_ngrams[:h]]
        i = 0
        while i < len(final_ngrams):
            delete = False
            for ngram in final_ngrams:
                if final_ngrams[i] != ngram and final_ngrams[i] in ngram:
                    delete = True
                    break
            if delete:
                del final_ngrams[i]
            else:
                i += 1
        final_ngrams= final_ngrams[:k]
        j=0
        for ngram in final_ngrams:
            if j<=k/len(obf_techniques) and ngram not in selected_features:
                selected_features.append(ngram)
                j+=1
    return selected_features


def extract_features(list data, list features, int feature_max_len, bool frequencies):
    cdef:
        int i
        int j
        string feature
        vector[string] features_vector = features
        cpp_map[string,float] extracted_features
        vector[vector[string]] data_vect

    for j in range(0,len(data),batch_size):
        sys.stdout.write('Reading batch from disk...\r')
        sys.stdout.flush()     
        data_vect.clear()
        for path in data[j:j+batch_size]:
            with open(path , 'rb') as f:
                data_vect.push_back(f.read().splitlines())
        sys.stdout.write(' '*80+'\r')
        sys.stdout.flush()

        with nogil, parallel(num_threads=n_core):
            for i in prange(data_vect.size(),schedule='dynamic'):
                extracted_features=extract_ngrams_app(data_vect[i],feature_max_len,frequencies)
                with gil:
                    data[j+i]=[extracted_features[feature] for feature in features_vector]
                    sys.stdout.write('{}/{} apps\r'.format(j+i+1,len(data)))
                    sys.stdout.flush()
    return data
