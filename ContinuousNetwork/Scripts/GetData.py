import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import random

def loadData():
    
    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir+"/DataSets")
    
    both_ends_360_sol = open("both_ends.txt", "r") #both-ends, upper and lower
    trajectoriesload_b_360 = np.loadtxt(both_ends_360_sol)

    trajectories_train = trajectoriesload_b_360
    trajectories_test = trajectories_train 
    
    num_nodes = trajectories_train.shape[1]//4
    os.chdir(original_dir)
    return num_nodes, trajectories_train, trajectories_test

def getDataLoaders(batch_size, number_elements,number_samples, number_samples_test, trajectories_train, trajectories_test, percentage_train):

    # Prepare the dictionaries where the data is stored
    x_full_train = np.concatenate((trajectories_train[:,:4],trajectories_train[:,-4:]),axis=1)
    y_full_train = trajectories_train[:,4:-4]
    x_full_test = np.concatenate((trajectories_test[:,:4],trajectories_test[:,-4:]),axis=1)
    y_full_test = trajectories_test[:,4:-4]
    Ntrain = int(percentage_train*number_samples)

    fact = 0.1
    
    x_train, y_train = x_full_train[:Ntrain], y_full_train[:Ntrain]
    if percentage_train == 0.8:
        Number_Test_Points = int(fact*number_samples)
        x_test, y_test = x_full_test[Ntrain:Ntrain+Number_Test_Points], y_full_test[Ntrain:Ntrain+Number_Test_Points]
        x_val, y_val = x_full_test[Ntrain+Number_Test_Points:Ntrain+2*Number_Test_Points], y_full_test[Ntrain+Number_Test_Points:Ntrain+2*Number_Test_Points] 
    else:
        Number_Test_Points = int(fact*number_samples)
        x_test, y_test = x_full_test[Ntrain:Ntrain+Number_Test_Points], y_full_test[Ntrain:Ntrain+Number_Test_Points]
        x_val, y_val = x_full_test[Ntrain+Number_Test_Points:Ntrain+2*Number_Test_Points], y_full_test[Ntrain+Number_Test_Points:Ntrain+2*Number_Test_Points] 
    
#     if percentage_train == 0.9:
#         x_train,x_test = x_full_train[:Ntrain], x_full_test[Ntrain:]
#         y_train,y_test = y_full_train[:Ntrain], y_full_test[Ntrain:]
#     else:
#         x_train,x_test = x_full_train[:Ntrain], x_full_test[Ntrain:Ntrain+int(0.1*number_samples)]
#         y_train,y_test = y_full_train[:Ntrain], y_full_test[Ntrain:Ntrain+int(0.1*number_samples)]  

    data_train = {
        "q1":[],
        "q2":[],
        "v1":[],
        "v2":[],
        "s":[],
        "sample_number":[],
        "qs":[],
        "vs":[]
    }

    data_test = {
        "q1":[],
        "q2":[],
        "v1":[],
        "v2":[],
        "s":[],
        "sample_number":[],
        "qs":[],
        "vs":[]
    }

    data_val = {
        "q1":[],
        "q2":[],
        "v1":[],
        "v2":[],
        "s":[],
        "sample_number":[],
        "qs":[],
        "vs":[]
    }
        
    #We consider the bijection of the interval [0,1] with the beam
    x_range = np.linspace(0,1,number_elements+1)

    list_nodes_boundary1 = np.array([0,1,2,3,5])
    list_nodes_boundary2 = np.array([number_elements-4,
                            number_elements-3,
                            number_elements-2,
                            number_elements-1,
                            number_elements])
    list_nodes_others = np.arange(6,number_elements-4,2)
    
    list_nodes = np.concatenate((list_nodes_boundary1, list_nodes_others, list_nodes_boundary2))
    
    for sample in range(number_samples):
        for node in list_nodes:
            if sample < Ntrain:
                #Spatial coordinate
                data_train["s"].append(x_range[node])
                data_train["sample_number"].append(sample)

                #Boundary conditions
                data_train["q1"].append(trajectories_train[sample][0:2])
                data_train["v1"].append(trajectories_train[sample][2:4])

                data_train["q2"].append(trajectories_train[sample][-4:-2])
                data_train["v2"].append(trajectories_train[sample][-2:])

                #True positions in correspondence to coordinate s
                data_train["qs"].append(trajectories_train[sample][4*node:4*node+2])
                data_train["vs"].append(trajectories_train[sample][4*node+2:4*node+4])
                
            if sample in range(Ntrain,Ntrain+Number_Test_Points):
                #Spatial coordinate
                data_test["s"].append(x_range[node])
                data_test["sample_number"].append(sample)

                #Boundary conditions
                data_test["q1"].append(trajectories_test[sample][0:2])
                data_test["v1"].append(trajectories_test[sample][2:4])

                data_test["q2"].append(trajectories_test[sample][-4:-2])
                data_test["v2"].append(trajectories_test[sample][-2:])

                #True positions in correspondence to coordinate s
                data_test["qs"].append(trajectories_test[sample][4*node:4*node+2])
                data_test["vs"].append(trajectories_test[sample][4*node+2:4*node+4])
            else:
                #Spatial coordinate
                data_val["s"].append(x_range[node])
                data_val["sample_number"].append(sample)

                #Boundary conditions
                data_val["q1"].append(trajectories_test[sample][0:2])
                data_val["v1"].append(trajectories_test[sample][2:4])

                data_val["q2"].append(trajectories_test[sample][-4:-2])
                data_val["v2"].append(trajectories_test[sample][-2:])

                #True positions in correspondence to coordinate s
                data_val["qs"].append(trajectories_test[sample][4*node:4*node+2])
                data_val["vs"].append(trajectories_test[sample][4*node+2:4*node+4])

    #Convert the dictionaries into numpy arrays
    data_train["q1"] = np.array(data_train["q1"])
    data_train["q2"] = np.array(data_train["q2"])
    data_train["v1"] = np.array(data_train["v1"])
    data_train["v2"] = np.array(data_train["v2"])
    data_train["s"] = np.array(data_train["s"])
    data_train["sample_number"] = np.array(data_train["sample_number"])
    data_train["qs"] = np.array(data_train["qs"])
    data_train["vs"] = np.array(data_train["vs"])

    data_test["q1"] = np.array(data_test["q1"])
    data_test["q2"] = np.array(data_test["q2"])
    data_test["v1"] = np.array(data_test["v1"])
    data_test["v2"] = np.array(data_test["v2"])
    data_test["s"] = np.array(data_test["s"])
    data_test["sample_number"] = np.array(data_test["sample_number"])
    data_test["qs"] = np.array(data_test["qs"])
    data_test["vs"] = np.array(data_test["vs"])
    
    data_val["q1"] = np.array(data_val["q1"])
    data_val["q2"] = np.array(data_val["q2"])
    data_val["v1"] = np.array(data_val["v1"])
    data_val["v2"] = np.array(data_val["v2"])
    data_val["s"] = np.array(data_val["s"])
    data_val["sample_number"] = np.array(data_val["sample_number"])
    data_val["qs"] = np.array(data_val["qs"])
    data_val["vs"] = np.array(data_val["vs"])
    
    print("train : ",x_train.shape)
    print("val : ",x_val.shape)
    print("test : ",x_test.shape)
    
    trainset = dataset(data_train)
    testset = dataset(data_test)
    valset = dataset(data_val)
    
    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(testset,batch_size=len(x_test),shuffle=True)
    valloader = DataLoader(valset,batch_size=len(x_val),shuffle=True)
    
    return data_train, data_test, data_val, x_train, x_test, y_train, y_test, x_val, y_val, trainloader, testloader, valloader

class dataset(Dataset):
  def __init__(self,data):

    self.q1 = torch.from_numpy(data["q1"].astype(np.float32))
    self.q2 = torch.from_numpy(data["q2"].astype(np.float32))
    self.v1 = torch.from_numpy(data["v1"].astype(np.float32))
    self.v2 = torch.from_numpy(data["v2"].astype(np.float32))
    self.s = torch.from_numpy(data["s"].astype(np.float32)).unsqueeze(1)
    self.sample = torch.from_numpy(data["sample_number"].astype(np.float32))
    self.qs = torch.from_numpy(data["qs"].astype(np.float32))
    self.vs = torch.from_numpy(data["vs"].astype(np.float32))
    self.length = self.q1.shape[0]

  def __getitem__(self,idx):
    return self.q1[idx],self.q2[idx],self.v1[idx],self.v2[idx],self.s[idx],self.sample[idx],self.qs[idx],self.vs[idx]
  def __len__(self):
    return self.length