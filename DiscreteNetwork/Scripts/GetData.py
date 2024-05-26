import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

def loadData(datacase = 1):
    
    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir+"/DataSets")
    
    both_ends_360_sol = open("both_ends.txt", "r") #both-ends, upper and lower
    trajectoriesload_b_360 = np.loadtxt(both_ends_360_sol)

    right_end_360_sol = open("right_end.txt", "r") #right-end, upper and lower
    trajectoriesload_r_360 = np.loadtxt(right_end_360_sol)
         
    if datacase == 1: # train and test on the both-ends data set
        trajectories_train = trajectoriesload_b_360
        trajectories_test = trajectories_train
    elif datacase == 2: # train on the both-ends data set and test on the right-end data set (EXTRAPOLATION)
        trajectories_train = trajectoriesload_b_360 
        trajectories_test = trajectoriesload_r_360 
    elif datacase == 3: # train and test on the both-ends+right-end data set
        trajectories_train = np.concatenate((trajectoriesload_b_360, trajectoriesload_r_360), axis = 0)
        trajectories_test = trajectories_train
    else:
        print("Warning! Must be an integer between 1 and 3")
        
    num_nodes = trajectories_train.shape[1]//4
    os.chdir(original_dir)
    return num_nodes, trajectories_train, trajectories_test

class dataset(Dataset):
  def __init__(self,x,y):

    self.bcs = torch.from_numpy(x.astype(np.float32))
    self.internal_node_outputs = torch.from_numpy(y.astype(np.float32))
    self.length = x.shape[0]

  def __getitem__(self,idx):
    return self.bcs[idx], self.internal_node_outputs[idx]
  def __len__(self):
    return self.length


def getDataLoaders(batch_size, datacase, percentage_train):

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    _, data_train, data_test = loadData(datacase)
    x_full_train = np.concatenate((data_train[:,:4],data_train[:,-4:]),axis=1)
    y_full_train = data_train[:,4:-4]
    N = len(x_full_train)
    NTrain = int(percentage_train*N) 
    
    idx_shuffle_train = np.arange(N)
    random.shuffle(idx_shuffle_train)
        
    x_full_train = x_full_train[idx_shuffle_train]
    y_full_train = y_full_train[idx_shuffle_train]
    
    x_full_test = np.concatenate((data_test[:,:4],data_test[:,-4:]),axis=1)
    y_full_test = data_test[:,4:-4]
 
    x_full_test = x_full_test[idx_shuffle_train]
    y_full_test = y_full_test[idx_shuffle_train]
    
    fact = 0.1
    
    x_train, y_train = x_full_train[:NTrain], y_full_train[:NTrain]

    Number_Test_Points = int(fact*N)
    x_test, y_test = x_full_test[NTrain:NTrain+Number_Test_Points], y_full_test[NTrain:NTrain+Number_Test_Points]
    x_val, y_val = x_full_test[NTrain+Number_Test_Points:NTrain+2*Number_Test_Points], y_full_test[NTrain+Number_Test_Points:NTrain+2*Number_Test_Points] 
 
    print("train : ",x_train.shape)
    print("val : ",x_val.shape)
    print("test : ",x_test.shape)
    
    trainset = dataset(x_train,y_train)
    testset = dataset(x_test,y_test)
    valset = dataset(x_val,y_val)
    
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(testset,batch_size=len(x_test),shuffle=True)
    valloader = DataLoader(valset,batch_size=len(x_val),shuffle=True)
    
    return x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader