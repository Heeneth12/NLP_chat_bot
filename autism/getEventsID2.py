import os
import pandas as pd


import numpy as np
import math

# mne imports
import mne
from mne import io
from mne.datasets import sample


# EEGNet-specific imports
from EEGModels import EEGNet,EEGNet_old
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split 
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

#Checking Working Directory
os.getcwd()
import scipy.io


os.chdir("D:\\aaDataSet\\data\\SBJ1\\S01\\Train\\")
data = scipy.io.loadmat('trainData.mat')
data = data['trainData']
#print(type(data))


line = open(os.path.join(os.getcwd(), 'trainEvents.txt'), 'r').readlines()
indexInfo = {'1':[], '2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[]}

indexCounter = 0
for i in line:
    indexCounter = indexCounter + 1
    if i == '1\n':
        indexInfo['1'].append(indexCounter)
    elif i == '2\n':
        indexInfo['2'].append(indexCounter)
    elif i == '3\n':
        indexInfo['3'].append(indexCounter)
    elif i == '4\n':
        indexInfo['4'].append(indexCounter)
    elif i == '5\n':
        indexInfo['5'].append(indexCounter)
    elif i == '6\n':
        indexInfo['6'].append(indexCounter)
    elif i == '7\n':
        indexInfo['7'].append(indexCounter)
    elif i == '8\n':
        indexInfo['8'].append(indexCounter)
    
data2 = data[5,:]
#print(len(indexInfo['3']))
indexesOfThirdEvent = []
dicEvents = {}



# =============================================================================
# Created A Dictionary That will be SAving Signals from Channel 5 Event 3 
# =============================================================================
for i in indexInfo['3']:
    indexesOfThirdEvent.append(i)

dataset3 = pd.DataFrame() 
dataset4 = pd.DataFrame() 
tempDataset = pd.DataFrame() 
for k in indexesOfThirdEvent:
    
    #print("Value of K : " , k)
    #print("Value of data[5,:,k]" , data[5,:,k])
    dicEvents[k] = data[5,:,k]
    value = data[5,:,k]
    
    df = pd.DataFrame(value)
    series = pd.Series(value)
    series2 = pd.DataFrame([series.tolist()], columns=series.index)
    dataset4 = pd.concat([df, dataset4])

    dataset3 = dataset3.append(series2, ignore_index=True)
    #dataset3 = dataset3.fillna(0)
    #dataset3 = dataset3.astype(int)
dataset3.to_csv('CHannel5_Event_3', index=False)



# =============================================================================
# Doing THis for CHannel 1
# =============================================================================

# =============================================================================
# Created A Dictionary That will be SAving Signals from Channel 1 Event 3 
# =============================================================================

indexesOfThirdEvent = []
dicEvents = {}

for i in indexInfo['3']:
    indexesOfThirdEvent.append(i)

newDataSet3 = pd.DataFrame() 
newDataSet4 = pd.DataFrame() 
tempDataset = pd.DataFrame() 
for k in indexesOfThirdEvent:
    
    #print("Value of K : " , k)
    #print("Value of data[5,:,k]" , data[5,:,k])
    dicEvents[k] = data[1,:,k]
    value = data[1,:,k]
    
    df = pd.DataFrame(value)
    series = pd.Series(value)
    series2 = pd.DataFrame([series.tolist()], columns=series.index)
    newDataSet4 = pd.concat([df, newDataSet4])

    newDataSet3 = newDataSet3.append(series2, ignore_index=True)
    #newDataSet3 = newDataSet3.fillna(0)
    #newDataSet3 = newDataSet3.astype(int)
newDataSet3.to_csv('CHannel1_Event_3', index=False)


# =============================================================================
# Now Taking the NExt Stwp. In whici we will be combining dataset of Channel 1
# and dataset of Channel 5. To create New Dataframe
# =============================================================================
    

finalFinalDataset = pd.concat([dataset3, newDataSet3], ignore_index=True)


# =============================================================================
# Adding Labels 
# =============================================================================

labels3 = pd.DataFrame([1]* 200)
labels4 = pd.DataFrame([0]* 200)
finalLabel = pd.concat([labels3, labels4], ignore_index=True)

finalDatasetWithLabels = pd.concat([finalFinalDataset, finalLabel], axis=1)




print(type(finalDatasetWithLabels))


#Shuffeling The Data


finalDatasetWithLabels = finalDatasetWithLabels.sample(frac=1).reset_index(drop=True)

#print(finalDatasetWithLabels2.shape)
#finalLabel = finalLabel.to_numpy()


# =============================================================================
# Adding Train, Test & Split
# =============================================================================

data_features = finalDatasetWithLabels.iloc[:,:-1]
y = finalDatasetWithLabels.iloc[:,-1:]




X_train, X_test, Y_train, Y_test = train_test_split(data_features,y,test_size=0.33,random_state=42)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
Y_train = Y_train.to_numpy()
Y_test = Y_test.to_numpy()

# Reshaping the Data for Neural Input
# =============================================================================


#X_train = X_train.to_numpy().reshape(400,1,350)


 
 
 #finalLabel = pd.DataFrame(finalLabel)
 #finalLabel = finalLabel.to_numpy().reshape(400,1,1) 
 
print(type(finalLabel))
 
 
X_train      = data_features[0:200]
Y_train      = y[0:200]
X_validate   = data_features[200:350]
Y_validate   = y[200:350]
X_test       = data_features[350:]
Y_test       = y[350:]


 

print("X_Test: ", X_test)

############################# EEGNet portion ##################################

kernels, chans, samples = 1, 1, 350

# convert labels to one-hot encodings.

try:
    # convert labels to one-hot encodings.
    Y_train      = np_utils.to_categorical(Y_train)
    Y_validate   = np_utils.to_categorical(Y_validate)
    Y_test       = np_utils.to_categorical(Y_test)

except ValueError:  #raised if `y` is empty.
    pass

print("X_train before Reshape: ", X_train.shape)
print("X_test before Reshape: ", X_test.shape)

    
# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
#X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)

#X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)


#X_train = X_train.reshape(200,1,350)
#X_test = X_test.reshape(50,1,350)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
X_validate = X_validate.to_numpy()


X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)



print("X_train after Reshape: ", X_train.shape)
print("X_test after Reshape: ", X_test.shape)



print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print('X_test shape:', X_test.shape)


# =============================================================================
#    X_train shape: (100, 1, 350, 1)
#   100 train samples
#   50 test samples
#   X_test shape: (50, 1, 350, 1)
# =============================================================================



# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)


model = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')


model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

checkpointer = ModelCheckpoint(filepath='checkpoint.h5', verbose=1,
                               save_best_only=True)

# count number of parameters in the model
numParams    = model.count_params() 

class_weights = {0:1, 1:1}


#fittedModel = model.fit(X_train, Y_train, batch_size = 2, epochs = 350, 
#                       verbose = 2, validation_data=(X_validate, Y_validate),
#                       callbacks=[checkpointer])

###############################################################################
# make prediction on test set.
###############################################################################

#probs       = model.predict(X_test)
#preds       = probs.argmax(axis = -1)  
#acc         = np.mean(preds == Y_test.argmax(axis=-1))
#print("Classification accuracy: %f " % (acc))



############################# PyRiemann Portion ##############################
# code is taken from PyRiemann's ERP sample script, which is decoding in 
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg     = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train      = X_train.reshape(X_train.shape[0], chans, samples)
X_test       = X_test.reshape(X_test.shape[0], chans, samples)


print("Just Before the Fit Command")
print("I Got Goosbumps, Type of X_test: " , type(X_test))
print("I Got Goosbumps, Type of Y_train: " , type(Y_train))

print("I Got Goosbumps, Shape of X_test: " , X_test.shape)
print("I Got Goosbumps, Shape of Y_train: " , Y_train.shape)



# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format

#clf = Covariances('oas').fit_transform(X_train)

#clf = Covariances('oas')
clf.fit(X_train, Y_train.argmax(axis = -1))
preds_rg     = clf.predict(X_test)




# Printing the results
acc2         = np.mean(preds_rg == Y_test.argmax(axis = -1))
print("Classification accuracy xDAWN : %f " % (acc2))

# plot the confusion matrices for both classifiers
names        = ['audio left', 'audio right', 'vis left', 'vis right']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')

plt.figure(1)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis = -1), names, title = 'xDAWN + RG')


