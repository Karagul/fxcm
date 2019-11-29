import tensorboard
import tensorflow
import pickle
import numpy as np
import pandas as pd

def ReadInputData(WindowToFuture = 5):

    with open('./made_dataset/DatasetWithIndicator_pickle', 'rb') as f:
        X_v = pickle.load(f)
    dataset1 = np.array(X_v)

    print (dataset1[5])
    dataset = pd.read_csv('./made_dataset/complete_data')
    label = []
    for i, x in enumerate(dataset):
        label.append(Classify(x['BidClose'], dataset['BidClose'][i+WindowToFuture], x['AskClose'], dataset['AskClose'][i+WindowToFuture]))
    LABEL = np.array(label)
    pickle_out = open('./made_dataset/LabelWindow5', 'wb')
    pickle.dump(LABEL, pickle_out)
    pickle_out.close()

def model():
    pass


ReadInputData()