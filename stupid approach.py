import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
#from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import time
#pd.set_option('display.max_columns', None)
#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('max_colwidth', 20)

# import datev event log data
# which consists of a string with "case;event;time;DOMAIN;SUBSYSTEM;MICROSUBSYSTEM;EVENTTAX_5;EVENTTAX_6;EVENTTAX_7"
# case event and time are relevant and
from sklearn.neighbors import KNeighborsClassifier

eventlog = pd.read_csv('data_converted.csv', delimiter=';')
#eventlog = pd.read_csv('helpdesk_converted.csv', delimiter=';')
eventlog = eventlog.drop(columns=['time'])

#modify data (datanew = data + 1)for calculation of gaussian process later, because event 0 is going to be a filler/blank
#LATER this would be reversed if a system in practical use cases would be needed
column_names = eventlog.columns ##take columns of a typical event log except the case column and increment
for i in range(1, len(column_names)):
    eventlog[column_names[i]] += 1

maxeventseq = (eventlog.groupby(['case']).count()['event']).max()

#now Build sample arrays for each case: for each k events do this k - 1 times
case_groups = eventlog.groupby('case')

samples_to_map = [] #contains a sequence of all samples in the event log, each sample is a 2 dimensional array of rows of event, domains and is filled with the number 0 for each row missing
labels = [] #1 dimensional array

for case, group in case_groups:
    for n in range(len(group) - 1): #create n - 1 samples in this group
        sample = [] #[maxeventseq][len(group.columns)] at the end
        for i in range(n + 1): #use i to current n amount of events
            sample_row = []
            for r in range(1, len(group.columns)):
                sample_row.append(group.iloc[i, r])
            sample.append(sample_row)
        for i_fillUp in range(n + 1, maxeventseq): #fill n to < maxeventseq with blank/0 event  ###minus 1 in range because of the last max event seq is going to be the lable
            sample_row = []
            for r in range(1, len(group.columns)):
                sample_row.append(0)
            sample.append(sample_row)

        samples_to_map.append(sample)
        labels.append(group.iloc[n + 1, 1])

#map the samples arra from 3d to 2d by concating every sample
samples = [] #array of mapped samples
for sample in samples_to_map:
    mapped_sample = []
    for sample_row in sample:
        for sample_feature in sample_row:
            mapped_sample.append(sample_feature)
    samples.append(mapped_sample)

samples_np = np.array(samples, dtype=int)
labels_np = np.array(labels, dtype=int)

trainsize = 0.70
testsize = 0.30
#Classification happens here
Sample_train, Sample_test, feature_train, feature_test = train_test_split(samples_np,
                                                                          labels_np,
                                                                          train_size=trainsize,
                                                                          test_size=testsize,

                                                                          )

start = time.time()

#kernel = 1.0 * RBF(length_scale=1.0)## [1.0] ##kernel=kernel #ConstantKernel(1.0) * RBF(length_scale=1.0)
#kernel = 0.75 * RBF([0.75])
gpc = RandomForestClassifier().fit(Sample_train, feature_train) #RandomForestClassifier #DecisionTreeClassifier(max_depth=5) ##KNeighborsClassifier(3)
#gpc = GaussianProcessClassifier().fit(Sample_train, feature_train) #, n_jobs=4
print('gpc done')
stop = time.time()
duration = int(stop - start)
print(duration)

#print options
np.set_printoptions(threshold=np.inf)
# calculate
accuracy =gpc.score(Sample_test, feature_test) # gpc

print(accuracy)

# Confusion Matrix
gpc_predictions = gpc.predict(Sample_test) #gpc
cm = confusion_matrix(feature_test, gpc_predictions)
#print(cm)

result_cf = pd.DataFrame(cm)
result_ac_time = pd.DataFrame([[accuracy, duration, trainsize, testsize
                                #, kernel.k1
                                ]], columns=['accuracy', 'runtime_in_s', 'training', 'testing'
                                        # , 'kernel'
                                            ]
                              )
result_cf.to_csv('result_conf_m.csv', index=False, sep=';')
result_ac_time.to_csv('result_accur_time.csv', index=False, sep=";")
