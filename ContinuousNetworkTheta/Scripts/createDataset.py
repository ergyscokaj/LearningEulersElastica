import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

def loadData():
    
    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir+"/DataSets")
    
    both_ends_360_sol = open("both_ends.txt","r")
    trajectoriesload_b_360 = np.loadtxt(both_ends_360_sol)
    
    # trajectoriesload_b_360 = np.delete(trajectoriesload_b_360, [0, 1, 499, 500, 998], 0)
    #trajectoriesload_b_360 = np.delete(trajectoriesload_b_360, [0,1,2,267,497,498,499,500,501,502,997,998,999], 0)

    trajectories = trajectoriesload_b_360
    
    num_nodes = trajectories.shape[1]//4
    os.chdir(original_dir)
    return num_nodes,trajectories

def getData(number_elements,number_samples,trajectories):

    # Prepare the dictionaries where the data is stored
    
    
    x_full = np.concatenate((trajectories[:,:4],trajectories[:,-4:]),axis=1)
    y_full = trajectories[:,4:-4]
    Ntrain = int(0.9*number_samples) #try to use less datapoints
    x_train,x_test = x_full[:Ntrain], x_full[Ntrain:]
    y_train,y_test = y_full[:Ntrain], y_full[Ntrain:]

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

    #We consider the bijection of the interval [0,1] with the beam
    x_range = np.linspace(0,1,number_elements+1)

    it = 0

    #Select the desired components of the trajectories and store them in the dictionaries
    
    list_nodes_boundary1 = np.array([0,1,2,3,5])
    list_nodes_boundary2 = np.array([number_elements-4,
                            number_elements-3,
                            number_elements-2,
                            number_elements-1,
                            number_elements])
    list_nodes_others = np.arange(6,number_elements-4,2)
    #list_nodes_others = np.arange(6,number_elements-3,3) #this was for the downloaded notebooks, so plots in paper
    
    list_nodes = np.concatenate((list_nodes_boundary1, list_nodes_others, list_nodes_boundary2))
    
    for sample in range(number_samples):
        for node in list_nodes: #np.arange(number_elements+1):

            if sample < Ntrain:
                #Spatial coordinate
                data_train["s"].append(x_range[node])
                data_train["sample_number"].append(sample)

                #Boundary conditions
                data_train["q1"].append(trajectories[sample][0:2])
                data_train["v1"].append(trajectories[sample][2:4])

                data_train["q2"].append(trajectories[sample][-4:-2])
                data_train["v2"].append(trajectories[sample][-2:])

                #True positions in correspondence to coordinate s
                data_train["qs"].append(trajectories[sample][4*node:4*node+2])
                data_train["vs"].append(trajectories[sample][4*node+2:4*node+4])
            else:
                #Spatial coordinate
                data_test["s"].append(x_range[node])
                data_test["sample_number"].append(sample)

                #Boundary conditions
                data_test["q1"].append(trajectories[sample][0:2])
                data_test["v1"].append(trajectories[sample][2:4])

                data_test["q2"].append(trajectories[sample][-4:-2])
                data_test["v2"].append(trajectories[sample][-2:])

                #True positions in correspondence to coordinate s
                data_test["qs"].append(trajectories[sample][4*node:4*node+2])
                data_test["vs"].append(trajectories[sample][4*node+2:4*node+4])

            it = it + 1

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
    
    return data_train, data_test, x_train, x_test, y_train, y_test
    

def getDataPINN(number_elements,number_samples,trajectories):

    # Prepare the dictionaries where the data is stored
    
    
    x_full = np.concatenate((trajectories[:,:4],trajectories[:,-4:]),axis=1)
    y_full = trajectories[:,4:-4]
    Ntrain = int(0.9*number_samples) #try to use less datapoints
    x_train,x_test = x_full[:Ntrain], x_full[Ntrain:]
    y_train,y_test = y_full[:Ntrain], y_full[Ntrain:]

    data_train = {
        "q1":[],
        "q2":[],
        "v1":[],
        "v2":[]
    }

    data_test = {
        "q1":[],
        "q2":[],
        "v1":[],
        "v2":[]
    }

    #We consider the bijection of the interval [0,1] with the beam
    x_range = np.linspace(0,1,number_elements+1)

    it = 0

    #Select the desired components of the trajectories and store them in the dictionaries
    for sample in range(number_samples):
        if sample < Ntrain:
            #Boundary conditions
            data_train["q1"].append(trajectories[sample][0:2])
            data_train["v1"].append(trajectories[sample][2:4])

            data_train["q2"].append(trajectories[sample][-4:-2])
            data_train["v2"].append(trajectories[sample][-2:])
        else:
            #Boundary conditions
            data_test["q1"].append(trajectories[sample][0:2])
            data_test["v1"].append(trajectories[sample][2:4])

            data_test["q2"].append(trajectories[sample][-4:-2])
            data_test["v2"].append(trajectories[sample][-2:])

    #Convert the dictionaries into numpy arrays
    data_train["q1"] = np.array(data_train["q1"])
    data_train["q2"] = np.array(data_train["q2"])
    data_train["v1"] = np.array(data_train["v1"])
    data_train["v2"] = np.array(data_train["v2"])

    data_test["q1"] = np.array(data_test["q1"])
    data_test["q2"] = np.array(data_test["q2"])
    data_test["v1"] = np.array(data_test["v1"])
    data_test["v2"] = np.array(data_test["v2"])
    
    return data_train, data_test, x_train, x_test, y_train, y_test



class datasetPINN(Dataset):
  def __init__(self,data):

    self.q1 = torch.from_numpy(data["q1"].astype(np.float32))
    self.q2 = torch.from_numpy(data["q2"].astype(np.float32))
    self.v1 = torch.from_numpy(data["v1"].astype(np.float32))
    self.v2 = torch.from_numpy(data["v2"].astype(np.float32))
    self.length = self.q1.shape[0]

  def __getitem__(self,idx):
    return self.q1[idx],self.q2[idx],self.v1[idx],self.v2[idx]
  def __len__(self):
    return self.length



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

def getDataLoaders(batch_size, data_train, data_test, type='pinn'):

    if type=='pinn':
        trainset = datasetPINN(data_train)
        testset = datasetPINN(data_test)
    else:
        trainset = dataset(data_train)
        testset = dataset(data_test)

    trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)
    testloader = DataLoader(testset,batch_size=len(testset),shuffle=True)
    return trainloader, testloader