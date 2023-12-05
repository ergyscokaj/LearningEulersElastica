import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.func import jacfwd, vmap
from Scripts.Utils import getBCs
import torch.nn as nn
from itertools import chain
import seaborn as sns

sns.set_style("darkgrid")
sns.set(font = "Times New Roman")
sns.set_context("paper")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt_kws = {"rasterized": True}

def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

#This returns the approximation vector q(s) in R^2 associated to the boundary conditions (q1,q2,v1,v2)
#and correspondent to position s in the interval [0,1]

def eval_model(model,device,s,q1,q2,v1,v2):
    s_ = torch.tensor([[s]],dtype=torch.float32).to(device)
    q1 = torch.from_numpy(q1.astype(np.float32)).reshape(1,-1).to(device)
    q2 = torch.from_numpy(q2.astype(np.float32)).reshape(1,-1).to(device)
    v1 = torch.from_numpy(v1.astype(np.float32)).reshape(1,-1).to(device)
    v2 = torch.from_numpy(v2.astype(np.float32)).reshape(1,-1).to(device)
    return model(s_,q1,q2,v1,v2).detach().cpu().numpy()[0]

def eval_derivative_model(model,device,s,q1,q2,v1,v2):
    s_ = torch.tensor([[s]],dtype=torch.float32).to(device)
    q1 = torch.from_numpy(q1.astype(np.float32)).reshape(1,-1).to(device)
    q2 = torch.from_numpy(q2.astype(np.float32)).reshape(1,-1).to(device)
    v1 = torch.from_numpy(v1.astype(np.float32)).reshape(1,-1).to(device)
    v2 = torch.from_numpy(v2.astype(np.float32)).reshape(1,-1).to(device)

    return model.derivative(s_,q1,q2,v1,v2).detach().cpu().numpy().reshape(-1)


def plotTestResults(model,device,number_elements,number_components,x_train,x_test,y_train,y_test, num_nodes, percentage_train):

    criterion = nn.MSELoss()

    training_trajectories = np.concatenate((x_train[:,:4],y_train,x_train[:,-4:]),axis=1)
    test_trajectories = np.concatenate((x_test[:,:4],y_test,x_test[:,-4:]),axis=1)
    all_trajectories = np.concatenate((training_trajectories, test_trajectories), axis = 0)

    bcs = getBCs(test_trajectories)
    q1 = bcs["q1"]
    q2 = bcs["q2"]
    v1 = bcs["v1"]
    v2 = bcs["v2"]

    q1 = torch.from_numpy(bcs["q1"].astype(np.float32)).to(device)
    q2 = torch.from_numpy(bcs["q2"].astype(np.float32)).to(device)
    v1 = torch.from_numpy(bcs["v1"].astype(np.float32)).to(device)
    v2 = torch.from_numpy(bcs["v2"].astype(np.float32)).to(device)

    q_idx = flatten_chain([[i,i+1] for i in np.arange(0,number_components,4)]) #indices of the qs
    qp_idx = flatten_chain([[i+2,i+3] for i in np.arange(0,number_components,4)]) #indices of the qs

    xx = torch.linspace(0,1,number_elements+1).unsqueeze(1).repeat(len(q1),1).to(device)
    one = torch.ones((number_elements+1,1)).to(device)
    q1_augmented = torch.kron(q1,one)
    q2_augmented = torch.kron(q2,one)
    v1_augmented = torch.kron(v1,one)
    v2_augmented = torch.kron(v2,one)

    pred_test_q = model(xx,q1_augmented,q2_augmented,v1_augmented,v2_augmented).reshape(len(q1),-1).detach().cpu().numpy()
    pred_test_qp = model.derivative(xx,q1_augmented,q2_augmented,v1_augmented,v2_augmented).reshape(len(q1),-1).detach().cpu().numpy()
    pred_test_all = np.zeros_like(test_trajectories)
    pred_test_all[:, q_idx] = pred_test_q
    pred_test_all[:, qp_idx] = pred_test_qp
    print(f"Error over test trajectories: {np.mean((pred_test_all-test_trajectories)**2)}.")

    ## Calculating the error on the training and all trajectories

    bcs_train = getBCs(training_trajectories)
    q1_train = bcs_train["q1"]
    q2_train = bcs_train["q2"]
    v1_train = bcs_train["v1"]
    v2_train = bcs_train["v2"]

    xx_train = np.linspace(0, 1, number_elements+1)

    q1_train = torch.from_numpy(bcs_train["q1"].astype(np.float32)).to(device)
    q2_train = torch.from_numpy(bcs_train["q2"].astype(np.float32)).to(device)
    v1_train = torch.from_numpy(bcs_train["v1"].astype(np.float32)).to(device)
    v2_train = torch.from_numpy(bcs_train["v2"].astype(np.float32)).to(device)

    xx_train = torch.linspace(0,1,number_elements+1).unsqueeze(1).repeat(len(q1_train),1).to(device)
    one = torch.ones((number_elements+1,1)).to(device)
    q1_train_augmented = torch.kron(q1_train,one)
    q2_train_augmented = torch.kron(q2_train,one)
    v1_train_augmented = torch.kron(v1_train,one)
    v2_train_augmented = torch.kron(v2_train,one)

    pred_train_q = model(xx_train,q1_train_augmented,q2_train_augmented,v1_train_augmented,v2_train_augmented).reshape(len(q1_train),-1).detach().cpu().numpy()
    pred_train_qp = model.derivative(xx_train,q1_train_augmented,q2_train_augmented,v1_train_augmented,v2_train_augmented).reshape(len(q1_train),-1).detach().cpu().numpy()
    pred_train_all = np.zeros_like(training_trajectories)
    pred_train_all[:, q_idx] = pred_train_q
    pred_train_all[:, qp_idx] = pred_train_qp
    print(f"Error over training trajectories: {np.mean((pred_train_all-training_trajectories)**2)}.")

    bcs_all = getBCs(all_trajectories)
    q1_all = bcs_all["q1"]
    q2_all = bcs_all["q2"]
    v1_all = bcs_all["v1"]
    v2_all = bcs_all["v2"]

    xx_all = np.linspace(0, 1, number_elements+1)

    q1_all = torch.from_numpy(bcs_all["q1"].astype(np.float32)).to(device)
    q2_all = torch.from_numpy(bcs_all["q2"].astype(np.float32)).to(device)
    v1_all = torch.from_numpy(bcs_all["v1"].astype(np.float32)).to(device)
    v2_all = torch.from_numpy(bcs_all["v2"].astype(np.float32)).to(device)

    xx_all = torch.linspace(0,1,number_elements+1).unsqueeze(1).repeat(len(q1_all),1).to(device)
    one = torch.ones((number_elements+1,1)).to(device)
    q1_all_augmented = torch.kron(q1_all,one)
    q2_all_augmented = torch.kron(q2_all,one)
    v1_all_augmented = torch.kron(v1_all,one)
    v2_all_augmented = torch.kron(v2_all,one)


    pred_all_q = model(xx_all,q1_all_augmented,q2_all_augmented,v1_all_augmented,v2_all_augmented).reshape(len(q1_all),-1).detach().cpu().numpy()
    pred_all_qp = model.derivative(xx_all,q1_all_augmented,q2_all_augmented,v1_all_augmented,v2_all_augmented).reshape(len(q1_all),-1).detach().cpu().numpy()
    pred_overall_all = np.zeros_like(all_trajectories)
    pred_overall_all[:, q_idx] = pred_all_q
    pred_overall_all[:, qp_idx] = pred_all_qp
    print(f"Error over all trajectories: {np.mean((pred_overall_all-all_trajectories)**2)}.")
    
    if percentage_train == 0.9:
        
        fig1 = plt.figure(figsize=(20, 15))
        for j in range(len(test_trajectories)):
            q_x_true = test_trajectories[j,np.arange(0,number_components,4)]
            q_y_true = test_trajectories[j,np.arange(1,number_components,4)]
            if j==0:
                plt.plot(q_x_true,q_y_true, '-', linewidth = 3, color = 'k', label = 'True')
                plt.plot(pred_test_q[j,np.arange(0,int(number_components/2),2)],pred_test_q[j,np.arange(1,int(number_components/2),2)], '--d', markersize = 5, linewidth = 1.8, color = 'r', label = 'Predicted')
                plt.legend()
            elif j in np.arange(2,test_trajectories.shape[0],11): #else
                plt.plot(q_x_true,q_y_true,'-', linewidth = 3, color = 'k')
                plt.plot(pred_test_q[j,np.arange(0,int(number_components/2),2)],pred_test_q[j,np.arange(1,int(number_components/2),2)], '--d', markersize = 5, linewidth = 1.8, color = 'r')
                plt.legend()
            plt.xlabel(r"$q_x$", fontsize = "45")
            plt.ylabel(r"$q_y$", fontsize = "45")
            plt.tick_params(labelsize = "45")
            plt.legend(fontsize = "45", loc = 'center')
            plt.title(r"Comparison over test trajectories $(q_x, q_y)$", fontsize = "45")

        fig2 = plt.figure(figsize=(20, 15))
        for j in range(len(test_trajectories)):
            v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
            v_y_true = test_trajectories[j,np.arange(3,number_components,4)]
            if j==0:
                plt.scatter(v_x_true,v_y_true,  color = 'k', s = 90, label = 'True')
                plt.scatter(pred_test_qp[j,np.arange(0,int(number_components/2),2)], pred_test_qp[j,np.arange(1,int(number_components/2),2)], color = 'r', s = 30, label = 'Predicted')
                plt.legend()
            elif j in np.arange(2,test_trajectories.shape[0],11): #else
                plt.scatter(v_x_true,v_y_true, color = 'k', s = 90)
                plt.scatter(pred_test_qp[j,np.arange(0,int(number_components/2),2)], pred_test_qp[j,np.arange(1,int(number_components/2),2)], color = 'r', s = 30)
                plt.legend()
            plt.xlabel(r"$q^{\prime}_x$", fontsize = "45")
            plt.ylabel(r"$q^{\prime}_y$", fontsize = "45")
            plt.tick_params(labelsize = "45")
            plt.axis('equal')
            plt.title(r"Comparison over test trajectories $(q^{\prime}_x, q^{\prime}_y)$", fontsize = "45")
            plt.legend(fontsize = "45", loc = 'best')

        norms_q = np.zeros((len(pred_test_q), num_nodes))
        mean_q = np.zeros(num_nodes)
        norms_qp = np.zeros((len(pred_test_q), num_nodes))
        mean_qp = np.zeros(num_nodes)
        for i in range(len(pred_test_q)):
            for j in range(num_nodes):
                norms_q[i, j] = np.linalg.norm(pred_test_q[i, 2*j:2*j+2] - test_trajectories[i,4*j:4*j+2])
                mean_q[j] = np.mean(norms_q[:, j])
                norms_qp[i, j] = np.linalg.norm(pred_test_qp[i, 2*j:2*j+2] - test_trajectories[i,4*j+2:4*j+4])
                mean_qp[j] = np.mean(norms_qp[:, j])

        fig3 = plt.figure(figsize = ((20, 15)))
        plt.plot(np.linspace(0, 50, 51), mean_q, '-d', linewidth = 2, color = 'k', label = r"Error on $(q_x, q_y)$")
        plt.plot(np.linspace(0, 50, 51), mean_qp, '-d', linewidth = 2, color = 'r', label = r"Error on $(q^{\prime}_x, q^{\prime}_y)$")
        plt.xlabel(r"node $k$", fontsize = "45")
        plt.ylabel(r"Average norm of error", fontsize = "45")
    #     plt.yscale("log")
        plt.tick_params(labelsize = "45")
        plt.title(r"Mean error over test trajectories", fontsize = "45")
        plt.legend(fontsize = "45", loc = 'best')
        plt.show()

    return pred_all_q, pred_all_qp