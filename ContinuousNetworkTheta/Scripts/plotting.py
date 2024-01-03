import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
from Scripts.utils import getBCs, reconstruct_q
import matplotlib.pyplot as plt
from itertools import chain
import os

sns.set_style("darkgrid")
sns.set(font = "Times New Roman")
sns.set_context("paper")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt_kws = {"rasterized": True}

def eval_derivative_model(model,device,s,q1,q2,v1,v2):
    s_ = torch.tensor([[s]],dtype=torch.float32).to(s.device)
    q1 = torch.from_numpy(q1.astype(np.float32)).reshape(1,-1).to(s.device)
    q2 = torch.from_numpy(q2.astype(np.float32)).reshape(1,-1).to(s.device)
    v1 = torch.from_numpy(v1.astype(np.float32)).reshape(1,-1).to(s.device)
    v2 = torch.from_numpy(v2.astype(np.float32)).reshape(1,-1).to(s.device)

    res = model(s_,q1,q2,v1,v2).detach().cpu().numpy().reshape(-1)
    return res


def plotTestResults(model,L,device,number_elements,number_components,x_train,x_test,y_train,y_test):

    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir+"/ContinuousNetworkTheta/SavedPlots")

    test_trajectories = np.concatenate((x_test[:,:4],y_test,x_test[:,-4:]),axis=1)

    bcs = getBCs(test_trajectories)
    q1 = bcs["q1"]
    q2 = bcs["q2"]
    v1 = bcs["v1"]
    v2 = bcs["v2"]
    xx = np.linspace(0, 1, number_elements+1) * L

    idx = np.arange(number_elements+1)

    res_derivative = np.zeros((len(test_trajectories),2,len(xx)))
    for j in range(len(test_trajectories)):
        for i in range(len(xx)):
            res_derivative[j,:,i] = eval_derivative_model(model,device,xx[i],q1[j],q2[j],v1[j],v2[j])
            #theta[j,i] = eval_theta(model,device,xx[i],v1[j],v2[j])

    res = reconstruct_q(q1,q2,v1,v2,L,model,device)

    fig = plt.figure(figsize=(20, 15))
    for j in range(len(test_trajectories)):
        q_x_true = test_trajectories[j,np.arange(0,number_components,4)]
        q_y_true = test_trajectories[j,np.arange(1,number_components,4)]
        v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
        v_y_true = test_trajectories[j,np.arange(3,number_components,4)]
        if j==0:
            plt.plot(q_x_true,q_y_true,'-', linewidth = 3, color = 'k', label = 'True')
            plt.plot(res[j,0],res[j,1],'--d', markersize = 5, linewidth = 1.8, color = 'r', label = 'Predicted')
            plt.legend(fontsize = "45")
        else:
            if j in np.arange(1,len(test_trajectories),11):
              plt.plot(q_x_true,q_y_true,'-', linewidth = 3, color = 'k')
              plt.plot(res[j,0],res[j,1],'--d', markersize = 5, linewidth = 1.8, color = 'r')
              plt.legend(fontsize = "45")

        plt.xlabel(r"$q_x$",fontsize="45")
        plt.ylabel(r"$q_y$",fontsize="45")
        plt.title(r"Comparison over test trajectories $(q_x,q_y)$",fontsize="45")
        plt.tick_params(labelsize = "45")
    plt.savefig("qs_plot_BC.pdf",bbox_inches='tight')
    #plt.show();

    fig = plt.figure(figsize=(20, 15))
    plt.axis('equal')
    for j in range(len(test_trajectories)):
        q_x_true = test_trajectories[j,np.arange(0,number_components,4)]
        q_y_true = test_trajectories[j,np.arange(number_components,4)]
        v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
        v_y_true = test_trajectories[j,np.arange(3,number_components,4)]
        if j==0:
            plt.scatter(v_x_true,v_y_true,color = 'k', s = 90, label = 'True')
            plt.scatter(res_derivative[j,0],res_derivative[j,1],color = 'r', s = 30, label = 'Predicted')
            plt.legend(fontsize = "45")
        else:
            if j in np.arange(1,len(test_trajectories),11):
              plt.scatter(v_x_true,v_y_true,color = 'k', s = 90)
              plt.scatter(res_derivative[j,0],res_derivative[j,1],color = 'r', s = 30)
              plt.legend(fontsize = "45")

        plt.xlabel(r"$q_x'$",fontsize="45")
        plt.ylabel(r"$q_y'$",fontsize="45")
        plt.tick_params(labelsize = "45")
        plt.title(r"Comparison over test trajectories $(q_x',q_y')$",fontsize="45")
    plt.savefig("vs_plot_BC.pdf",bbox_inches='tight')
    #plt.show();

    fig = plt.figure(figsize=(20, 15))
    norm_v = 0.
    norm_q = 0.
    for j in range(len(test_trajectories)):
        q_x_true = test_trajectories[j,np.arange(0,number_components,4)]
        q_y_true = test_trajectories[j,np.arange(1,number_components,4)]
        v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
        v_y_true = test_trajectories[j,np.arange(3,number_components,4)]

        norm_v += np.sqrt((v_x_true-res_derivative[j,0])**2 + (v_y_true-res_derivative[j,1])**2)
        norm_q += np.sqrt((q_x_true-res[j,0])**2 + (q_y_true-res[j,1])**2)

    plt.plot(idx,norm_q /len(test_trajectories) ,'-d', linewidth = 2, color = 'k', label = r"Error on $(q_x, q_y)$ ")
    plt.plot(idx,norm_v / len(test_trajectories),'-d', linewidth = 2, color = 'r', label = r"Error on $(q^{\prime}_x, q^{\prime}_y)$")
    plt.legend(fontsize = "45")
    plt.xlabel(r"node $k$",fontsize="45")
    plt.ylabel(r"Average norm of error",fontsize="45")
    plt.title(r"Mean error over test trajectories",fontsize="45")
    plt.tick_params(labelsize = "45")
    plt.savefig("error_mean_BC.pdf",bbox_inches='tight')
    #plt.show();

    fig = plt.figure(figsize=(20, 15))
    for j in range(len(test_trajectories)):
      vv1 = torch.from_numpy(v1[j:j+1].astype(np.float32)).to(device)
      vv2 = torch.from_numpy(v2[j:j+1].astype(np.float32)).to(device)
      qq1 = torch.from_numpy(q1[j:j+1].astype(np.float32)).to(device)
      qq2 = torch.from_numpy(q2[j:j+1].astype(np.float32)).to(device)

      theta1 = torch.atan2(vv1[:,1:2],vv1[:,0:1]).detach().cpu().numpy()
      s = torch.from_numpy(xx.astype(np.float32)).unsqueeze(1).to(device)
      theta = model.theta(s,qq1.repeat(len(s),1),qq2.repeat(len(s),1),vv1.repeat(len(s),1),vv2.repeat(len(s),1))[:,0].detach().cpu().numpy()
      v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
      v_y_true = test_trajectories[j,np.arange(3,number_components,4)]

      theta_true = (np.arctan2(v_y_true,v_x_true))

      if j==0:
        plt.plot(xx,theta_true,'-', linewidth = 3, color = 'k', label = 'True')
        plt.plot(xx,theta,'--d', markersize = 5, linewidth = 1.8, color = 'r', label = 'Predicted')
      else:
        if j in np.arange(1,len(test_trajectories),11):
          plt.plot(xx,theta_true,'-', linewidth = 3, color = 'k')
          plt.plot(xx,theta,'--d', markersize = 5, linewidth = 1.8, color = 'r')

    plt.xlabel(r"$s$",fontsize="45")
    plt.ylabel(r"$\theta$",fontsize="45")
    plt.legend(fontsize = "45")
    plt.title(r"Comparison over test trajectories $\theta$",fontsize="45")
    plt.tick_params(labelsize = "45")
    plt.savefig("theta_plot_BC.pdf",bbox_inches='tight')
    #plt.show();

    fig = plt.figure(figsize=(20, 15))
    for j in range(len(test_trajectories)):
        v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
        v_y_true = test_trajectories[j,np.arange(3,number_components,4)]
        if j==0:
            plt.plot(xx,v_x_true,'-',linewidth = 0.8, color = 'C0', label = 'True')
            plt.plot(xx,res_derivative[j,0],'--',linewidth = 0.8, color = 'C1', label = 'Predicted')
        else:
           plt.plot(xx,v_x_true,'-',linewidth = 0.8, color = 'C0')
           plt.plot(xx,res_derivative[j,0],'--',linewidth = 0.8, color = 'C1')

        plt.legend(fontsize = "45")

        plt.xlabel(r"$s$", fontsize = "45")
        plt.ylabel(r"$q_x'$", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.title(r"Comparison over test trajectories : $q_x'$", fontsize = "45")
    plt.savefig("vx_plot_BC.pdf",bbox_inches='tight')
    #plt.show();

    fig = plt.figure(figsize=(20, 15))
    for j in range(len(test_trajectories)):
        v_x_true = test_trajectories[j,np.arange(2,number_components,4)]
        v_y_true = test_trajectories[j,np.arange(3,number_components,4)]
        if j==0:
            plt.plot(xx,v_y_true,'-',linewidth = 0.8, color = 'C0', label = 'True')
            plt.plot(xx,res_derivative[j,1],'--',linewidth = 0.8, color = 'C1', label = 'Predicted')
            plt.legend(fontsize = "45")
        else:
            plt.plot(xx,v_y_true,'-',linewidth = 0.8, color = 'C0')
            plt.plot(xx,res_derivative[j,1],'--',linewidth = 0.8, color = 'C1')
            plt.legend(fontsize = "45")

        plt.xlabel(r"$s$", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.ylabel(r"$q_y'$", fontsize = "45")
        plt.title(r"Comparison over test trajectories : $q_y'$", fontsize = "45")
    plt.savefig("vy_plot_BC.pdf",bbox_inches='tight')
    #plt.show();
    
    os.chdir(original_dir)

    return res_derivative,theta

def eval_derivative_model(model,device,s,q1,q2,v1,v2):
    s_ = torch.tensor([[s]],dtype=torch.float32).to(device)
    q1 = torch.from_numpy(q1.astype(np.float32)).reshape(1,-1).to(device)
    q2 = torch.from_numpy(q2.astype(np.float32)).reshape(1,-1).to(device)
    v1 = torch.from_numpy(v1.astype(np.float32)).reshape(1,-1).to(device)
    v2 = torch.from_numpy(v2.astype(np.float32)).reshape(1,-1).to(device)

    res = model(s_,q1,q2,v1,v2).detach().cpu().numpy().reshape(-1)
    return res

def flatten_chain(matrix):
    return list(chain.from_iterable(matrix))

def compute_errors(model,L,device,number_elements,number_components,x_train,x_test,y_train,y_test):

    v_idx = flatten_chain([[i,i+1] for i in np.arange(2,number_components,4)]) #indices of the vs
    q_idx = flatten_chain([[i,i+1] for i in np.arange(0,number_components,4)]) #indices of the qs


    #Test trajectories
    test_trajectories = np.concatenate((x_test[:,:4],y_test,x_test[:,-4:]),axis=1)
    bcs = getBCs(test_trajectories)
    q1 = bcs["q1"]
    q2 = bcs["q2"]
    v1 = bcs["v1"]
    v2 = bcs["v2"]
    xx = np.linspace(0, 1, number_elements+1) * L

    idx = np.arange(number_elements+1)

    res_derivative_test = np.zeros((len(test_trajectories),2,len(xx)))
    for j in range(len(test_trajectories)):
        for i in range(len(xx)):
            res_derivative_test[j,:,i] = eval_derivative_model(model,device,xx[i],q1[j],q2[j],v1[j],v2[j])

    res_test = reconstruct_q(q1,q2,v1,v2,L,model,device)

    #Training trajectories
    training_trajectories = np.concatenate((x_train[:,:4],y_train,x_train[:,-4:]),axis=1)
    bcs = getBCs(training_trajectories)
    q1 = bcs["q1"]
    q2 = bcs["q2"]
    v1 = bcs["v1"]
    v2 = bcs["v2"]
    xx = np.linspace(0, 1, number_elements+1) * L

    idx = np.arange(number_elements+1)

    res_derivative_train = np.zeros((len(training_trajectories),2,len(xx)))
    for j in range(len(training_trajectories)):
        for i in range(len(xx)):
            res_derivative_train[j,:,i] = eval_derivative_model(model,device,xx[i],q1[j],q2[j],v1[j],v2[j])

    res_train = reconstruct_q(q1,q2,v1,v2,L,model,device)

    #Compute errors
    q_train_pred = res_train.transpose(0,2,1).reshape(len(res_train),-1)
    q_test_pred = res_test.transpose(0,2,1).reshape(len(res_test),-1)
    v_train_pred = res_derivative_train.transpose(0,2,1).reshape(len(res_derivative_train),-1)
    v_test_pred = res_derivative_test.transpose(0,2,1).reshape(len(res_derivative_test),-1)

    pred_train_all = np.zeros_like(training_trajectories)
    pred_test_all = np.zeros_like(test_trajectories)

    pred_train_all[:, q_idx] = q_train_pred
    pred_test_all[:, q_idx] = q_test_pred
    pred_train_all[:, v_idx] = v_train_pred
    pred_test_all[:, v_idx] = v_test_pred

    test_error = np.mean((pred_train_all-training_trajectories)**2)
    train_error = np.mean((pred_test_all-test_trajectories)**2)

    return test_error, train_error, pred_train_all, pred_test_all