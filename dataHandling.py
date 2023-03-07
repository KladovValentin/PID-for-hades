import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
import uproot
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


class My_dataset(Dataset):
    def __init__(self, dataTable):
        self.datasetX, self.datasetY = dataTable[0], dataTable[1]

    def __len__(self):
        return len(self.datasetY)

    def __getitem__(self, index):
        return torch.tensor(self.datasetX[index]), torch.tensor(self.datasetY[index])


def load_dataset(dataTable):
    # transform to numpy, assign types, split on features-labels
    df = dataTable
    dfn = df.to_numpy()

    x = dfn[:,:-1].astype(np.float32)
    y = dfn[:, -1].astype(int)

    print('x shape = ' + str(x.shape))
    print('y shape = ' + str(y.shape))
    return (x, y)



class DataManager():
    def __init__(self) -> None:
        self.poorColumnValues = [('tofdedx',-1)]
    

    def prepareTable(self, datF):
        # make datasets equal sizes for each class out of n
        lastName = list(datF.columns)[-1]
        nClasses = datF[lastName].nunique()
        x = [len(datF[(datF[lastName]==i)]) for i in range(nClasses)]
        print("classes lenghts = : " + str(x))
        minimumCount = np.amin(np.array(x))
        frames = [datF.loc[datF[lastName] == i].sample(minimumCount) for i in range(nClasses)]
        print("new classes lenghts = : " + str([len(frames[i]) for i in range(nClasses)]))
        return pandas.concat(frames).sort_index().reset_index(drop=True)
        #print(datF)
        #return datF.sort_index().reset_index(drop=True)


    def meanAndStdTable(self, dataTable):
        # find mean and std values for each column of the dataset (used for train dataset)
        df = dataTable
        dfn = df.to_numpy()

        x = dfn[:,:].astype(np.float32)
        mean = np.array( [np.mean(x[:,j]) for j in range(x.shape[1])] )
        std  = np.array( [np.std( x[:,j]) for j in range(x.shape[1])] )
        
        #__ if you have bad data sometimes in one of the columns - 
        # - you can calculate mean and std without these bad entries
        #   and then make them = 0 -> no effect on the first layer
        for i in range(len(self.poorColumnValues)):
            cPoor = df.columns.get_loc(self.poorColumnValues[i][0])
            vPoor = self.poorColumnValues[i][1]
            mean[cPoor] = np.mean(x[(x[:,cPoor]!=vPoor),cPoor])
            std[cPoor] = np.std(x[(x[:,cPoor]!=vPoor),cPoor])

        return mean, std


    def normalizeDataset(self, df, meanValues, stdValues):
        columns = list(df.columns)
        masks = []
        for i in range(len(self.poorColumnValues)):
            masks.append(df[self.poorColumnValues[i][0]]==self.poorColumnValues[i][1])

        for i in range(len(columns)-1):
            df[columns[i]] = (df[columns[i]]-meanValues[i])/stdValues[i]
        
        for i in range(len(self.poorColumnValues)):
            df[self.poorColumnValues[i][0]].mask(masks[i], 0, inplace=True)
        return df


    def getDataset(self, rootPath,mod):
        # read data, select raws (pids) and columns (drop)

        tables = []
        for batch in uproot.iterate([rootPath],library="pd"):
            tables.append(batch)
        setTable = pandas.concat(tables).sort_index().reset_index(drop=True)
        selection = (setTable['mass2']>-0.5) & (setTable['mass2']<2.5) & (setTable['momentum']>0.05) & (setTable['momentum']<5) & (setTable['mdcdedx']>0.1) & (setTable['mdcdedx']<30)
        setTable = setTable.loc[selection].copy()

        if mod == "simLabel":
            pidsToSelect = [8,11,14]
            ttables = []
            for i in range(len(pidsToSelect)):
                ttables.append(setTable.loc[setTable['pid']==pidsToSelect[i]].copy())
                ttables[i]['pid'] = i
            fullSetTable = pandas.concat(ttables).sort_index()

            fullSetTableBad = setTable.drop(fullSetTable.index)
            fullSetTableBad['pid'] = len(pidsToSelect)
            setTable = pandas.concat([fullSetTable]).sort_index().reset_index(drop=True)

        dropColumns = ['beta','ringcorr','event_id', 'theta', 'phi']
        for drop in dropColumns:
            setTable.drop(drop,axis=1,inplace=True)

        return setTable


    def manageDataset(self, mod):
        dftCorr = self.prepareTable(self.getDataset("sim/*sim*.root:pid", "simLabel"))
        setTable = self.getDataset("data/*data*.root:pid", "data").sample(frac=0.5).sort_index().reset_index(drop=True)

        mean, std = 0, 0
        if (mod == "train_dann"):
            mean, std = self.meanAndStdTable(pandas.concat([dftCorr,setTable], ignore_index=True))
        elif (mod == "train_nn"):
            mean, std = self.meanAndStdTable(dftCorr,setTable)
        elif (mod.startswith("test")):
            mean, std = readTrainData()

        dftCorr = self.normalizeDataset(dftCorr,mean,std).copy()
        setTable = self.normalizeDataset(setTable,mean,std)
        print(dftCorr)

        pq.write_table(pa.Table.from_pandas(dftCorr), 'simu1.parquet')
        pq.write_table(pa.Table.from_pandas(setTable), 'expu1.parquet')

        if (mod.startswith("train")):
            writeTrainData(mean,std)



def writeTrainData(meanArr,stdArr):
    np.savetxt('meanValues.txt', meanArr, fmt='%s')
    np.savetxt('stdValues.txt', stdArr, fmt='%s')

def readTrainData():
    meanValues = np.loadtxt('meanValues.txt')
    stdValues = np.loadtxt('stdValues.txt')
    return meanValues, stdValues

