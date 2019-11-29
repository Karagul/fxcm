import pandas as pd
import pickle
import numpy as np
import datetime
from collections import deque
import statistics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import deque

def Classify(currentBid, futireBid, currentAsk, futureAsk):

    # print (currentBid, futireBid, currentAsk, futureAsk)
    if float(futireBid) > float(currentAsk):
        return [1,0,0]  #buy, dont do anything, sell
    elif float(currentBid) > float(futureAsk):
        return [0,0,1]
    else:
        return[0,1,0]


def put_to_gether ():

    for j in range(2012, 2019):
        for i in range (1,53):      # it goes from 1 to 52
            name = './dataset/m1_EURUSD_'+str(j)+'_'+str(i)
            if name == './dataset/m1_EURUSD_2012_1':
                df = pd.read_csv(name)
                df.to_csv('./made_dataset/complete_data')
                continue

            print (name)
            df = pd.read_csv(name)
            df.to_csv('./made_dataset/complete_data', mode='a', header=False)
            dataset = pd.read_csv('./made_dataset/complete_data') # read the file that was made now
            dataset.to_pickle('./made_dataset/complete_data_pickle')  # wtite a file in pickle

def indicators(WindowSize = 30, alpha = 0.1, WindowToFuture = 3):

    PastWindowBin = deque(maxlen=WindowSize)
    PastWindowAsk = deque(maxlen=WindowSize)

    Past26Ask = deque(maxlen=26)
    Past12Ask = deque(maxlen=12)
    Past26Bid = deque(maxlen=26)
    Past12Bid = deque(maxlen=12)

    Past20Ask = deque(maxlen=20)
    Past20Bid = deque(maxlen=20)

    Past14GainAsk = deque(maxlen=14)
    Past14GainBid = deque(maxlen=14)
    Past14LossAsk = deque(maxlen=14)
    Past14LossBid = deque(maxlen=14)

    dataset = pd.read_csv('./made_dataset/complete_data')

    label = []
    SMABid_l = []
    SMAAsk_l = []
    EMABid_l = []
    EMAAsk_l = []
    MACDBid_l = []
    MACDAsk_l = []
    BBBidMid_l = []
    BBBidLow_l = []
    BBBidHigh_l = []
    BBAskMid_l = []
    BBAskLow_l = []
    BBAskHigh_l = []
    RSIBid_l = []
    RSIAsk_l = []

    EMABid = 0
    EMAAsk = 0
    # df = pd.DataFrame(dataset)
    for i, x in enumerate(dataset['DateTime']):
        try:

            label.append(Classify(dataset['BidClose'][i], dataset['BidClose'][i + WindowToFuture], dataset['AskClose'][i],
                          dataset['AskClose'][i + WindowToFuture]))
        except:
            label.append(0)

        try:
            PastWindowAsk.appendleft(dataset['AskClose'][i])
            PastWindowBin.appendleft(dataset['BidClose'][i])
            if len(PastWindowAsk) >=WindowSize:
                SMABid = sum([k for k in PastWindowBin]) / WindowSize  #
                SMABid_l.append(SMABid)
                Past20Bid.appendleft(SMABid)
                SMAAsk = sum([k for k in PastWindowAsk]) / WindowSize  #
                SMAAsk_l.append(SMAAsk)
                Past20Ask.appendleft(SMAAsk)
            else:
                SMABid = 0
                SMABid_l.append(SMABid)
                SMAAsk = 0
                SMAAsk_l.append(SMAAsk)
        except:
            SMABid = 99999
            SMABid_l.append(SMABid)
            SMAAsk = 99999
            SMAAsk_l.append(SMAAsk)

        EMABid = alpha * dataset['BidClose'][i] + (1 - alpha) * EMABid  #
        EMABid_l.append(EMABid)
        Past26Bid.appendleft(EMABid)
        Past12Bid.appendleft(EMABid)
        EMAAsk = alpha * dataset['AskClose'][i] + (1 - alpha) * EMAAsk  #
        EMAAsk_l.append(EMAAsk)
        Past26Ask.appendleft(EMAAsk)
        Past12Ask.appendleft(EMAAsk)

        try:
            Dum = dataset['BidClose'][i] - dataset['BidClose'][i - 1]
            if Dum>=0:
                Past14GainBid.appendleft(100*Dum/dataset['BidClose'][i - 1])
                Past14LossBid.appendleft(0)
            else:
                Past14LossBid.appendleft(100*Dum/dataset['BidClose'][i - 1])
                Past14GainBid.appendleft(0)
        except:
            print('problem with gain Bid')

        try:
            Dum = dataset['AskClose'][i] - dataset['AskClose'][i - 1]
            if Dum>=0:
                Past14GainAsk.appendleft(100*Dum/dataset['AskClose'][i - 1])
                Past14LossAsk.appendleft(0)
            else:
                Past14LossAsk.appendleft(100*Dum/dataset['AskClose'][i - 1])
                Past14GainAsk.appendleft(0)
        except:
            print('problem with gain Ask')

        if i >=90:
            MACDBid = (sum([k for k in Past12Bid]) /12) - (sum([k for k in Past26Bid]) /26) #
            MACDBid_l.append(MACDBid)
            MACDAsk = (sum([k for k in Past12Ask]) /12) - (sum([k for k in Past26Ask]) /26) #
            MACDAsk_l.append(MACDAsk)

            BBBidMid =  (sum([k for k in Past20Bid]) /20) #
            BBBidLow = (sum([k for k in Past20Bid]) /20) - 2*(statistics.stdev([k for k in Past20Bid]))#
            BBBidHigh = (sum([k for k in Past20Bid]) /20) + 2*(statistics.stdev([k for k in Past20Bid]))#
            BBBidMid_l.append(BBBidMid)
            BBBidLow_l.append(BBBidLow)
            BBBidHigh_l.append(BBBidHigh)
            BBAskMid = (sum([k for k in Past20Ask]) /20) #
            BBAskLow = (sum([k for k in Past20Ask]) /20) - 2*(statistics.stdev([k for k in Past20Ask]))#
            BBAskHigh = (sum([k for k in Past20Ask]) /20) + 2*(statistics.stdev([k for k in Past20Ask]))#
            BBAskMid_l.append(BBAskMid)
            BBAskLow_l.append(BBAskLow)
            BBAskHigh_l.append(BBAskHigh)

            # print (sum([k for k in Past14LossBid]))
            try:
                RSIBid = 100 - (100/(1+(sum([k for k in Past14GainBid])/sum([k for k in Past14LossBid]))))  #
                RSIAsk = 100 - (100/(1+(sum([k for k in Past14GainAsk])/sum([k for k in Past14LossAsk]))))  #
                RSIBid_l.append(RSIBid)
                RSIAsk_l.append(RSIAsk)
            except:
                RSIBid_l.append(0)
                RSIAsk_l.append(0)

        else:
            MACDBid = 0
            MACDAsk = 0

            BBBidMid = 0
            BBBidLow =0
            BBBidHigh =0
            BBAskMid =0
            BBAskLow =0
            BBAskHigh =0
            RSIBid =0
            RSIAsk =0
            MACDAsk_l.append(MACDAsk)
            MACDBid_l.append(MACDBid)
            RSIBid_l.append(RSIBid)
            RSIAsk_l.append(RSIAsk)
            BBBidMid_l.append(BBBidMid)
            BBBidLow_l.append(BBBidLow)
            BBBidHigh_l.append(BBBidHigh)
            BBAskMid_l.append(BBAskMid)
            BBAskLow_l.append(BBAskLow)
            BBAskHigh_l.append(BBAskHigh)
            print ('not yet 26 long')

        if i%100000 == 0:
            print('1000 more data processed', i)

    # putthing the data in the dataset
    dataset['SMABid'] = SMABid
    dataset['SMAAsk'] = SMAAsk
    dataset['EMABid'] = EMABid
    dataset['EMAAsk'] = EMAAsk
    dataset['MACDBid'] = MACDBid
    dataset['MACDAsk'] = MACDAsk
    dataset['BBBidMid'] = BBBidMid
    dataset['BBBidLow'] =BBBidLow
    dataset['BBBidHigh'] =BBBidHigh
    dataset['BBAskMid'] =BBAskMid
    dataset['BBAskLow'] =BBAskLow
    dataset['BBAskHigh'] =BBAskHigh
    dataset['RSIBid'] =RSIBid
    dataset['RSIAsk'] =RSIAsk

    dataset['target'] = label

    dataset.to_csv('./made_dataset/DatasetWithIndictor_with_head', mode='a')
    dataset.to_pickle('./made_dataset/DatasetWithIndicator_pickle_with_head')  # wtite a file in pickle

def windowed(WindowSize=30):
    dataset = pd.read_csv('./made_dataset/DatasetWithIndictor_with_head')
    # print(dataset.head())
    array = np.array(dataset)
    # print (array.shape, array[1,:])
    # print ([k for k in dataset.columns])
    SequantialData = []

    min_max_scaler = MinMaxScaler()
    normal_dataset = min_max_scaler.fit_transform(array[:,3:-1])
    # print (normal_dataset.shape, normal_dataset[1,:])

    pre_mins = deque(maxlen=WindowSize)
    for i in range(len(normal_dataset)):
        pre_mins.append([n for n in normal_dataset[i,:]])
        # print (pre_mins)
        if len(pre_mins) == WindowSize and i > 60:
            SequantialData.append((np.array(pre_mins), np.array(dataset['target'][i])))
            # print (SequantialData)
        if i%100000 == 0:
            print ('10000 processed', i)

    df_train, df_test = train_test_split(SequantialData, train_size=0.92, test_size=0.08, shuffle=False)
    print("Train and Test size", len(df_train), len(df_test))

    # dataset.to_csv('./made_dataset/DATASET_READY_train', mode='a')
    dataset.to_pickle('./made_dataset/DATASET_READY_Train_pickle')  # wtite a file in pickle
    dataset.to_pickle('./made_dataset/DATASET_READY_Val_pickle')  # wtite a file in pickle





def main():
    put_to_gether() #make the csv file with all the prices
    indicators() # calculate the indicators and put them into another file with the features
    windowed()

if __name__ == '__main__':
    main()