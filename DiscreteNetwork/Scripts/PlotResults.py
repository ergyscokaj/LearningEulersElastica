import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from Scripts.GetData import getDataLoaders, loadData
from Scripts.Training import train
from csv import writer
import time

sns.set_style("darkgrid")
sns.set(font = "Times New Roman")
sns.set_context("paper")
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt_kws = {"rasterized": True}

def plotResults(model, device, x_train, y_train, x_test, y_test, x_val, y_val, num_nodes, datacase, percentage_train, gamma, number_layers, hidden_nodes):

    train_bcs = torch.from_numpy(x_train.astype(np.float32)).to(device)
    test_bcs = torch.from_numpy(x_test.astype(np.float32)).to(device)
    val_bcs = torch.from_numpy(x_val.astype(np.float32)).to(device)
    
    pred_train = np.concatenate((x_train[:, :4], model(train_bcs).detach().cpu().numpy(), x_train[:, -4:]), axis = 1)
    pred_test = np.concatenate((x_test[:, :4], model(test_bcs).detach().cpu().numpy(), x_test[:, -4:]), axis = 1)
    pred_val = np.concatenate((x_val[:, :4], model(val_bcs).detach().cpu().numpy(), x_val[:, -4:]), axis = 1)

    true_train = np.concatenate((x_train[:, :4], y_train, x_train[:, -4:]), axis = 1)
    true_test = np.concatenate((x_test[:, :4], y_test, x_test[:, -4:]), axis = 1)
    true_val = np.concatenate((x_val[:, :4], y_val, x_val[:, -4:]), axis = 1)
    
    pred = np.concatenate((pred_train[:,4:-4], pred_test[:,4:-4], pred_val[:,4:-4]), axis = 0)
    true = np.concatenate((true_train[:,4:-4], true_test[:,4:-4], true_val[:,4:-4]), axis = 0)
    error_all = np.mean((pred - true)**2)
    error_training = np.mean((pred_train[:,4:-4] - true_train[:,4:-4])**2)
    error_testing = np.mean((pred_test[:,4:-4] - true_test[:,4:-4])**2)
    error_validation = np.mean((pred_val[:,4:-4] - true_val[:,4:-4])**2)
    
    test_bvs = torch.from_numpy(x_test.astype(np.float32))
    initial_time = time.time()
    _ = model(test_bvs)
    final_time = time.time()
    total_time = final_time-initial_time
    print("Number of trajectories in the test set : ",len(test_bvs))
    print("Total time to predict test trajectories : ",total_time)
    print("Average time to predict test trajectories : ",total_time / len(test_bvs))

    labels = ["pctg_train","datacase","gamma","nlayers","hidden_nodes","training_error","testing_error","validation_error","total_error"]
    
    results = [str(percentage_train),str(datacase),str(gamma), str(number_layers), str(hidden_nodes), str(error_training),str(error_testing),str(error_validation),str(error_all)]
    
    with open(f"results_train_val_test_splits.csv", "a") as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(labels)
        writer_object.writerow(results)
        f_object.close()

    print(f"\n Error over training trajectories: {error_training}, \n Error over test trajectories: {error_testing}, \n Error over validation trajectories: {error_validation}, \n Error over all trajectories: {error_all}.")

    c, d = pred_test.shape

    norms_q = np.zeros((len(pred_test), num_nodes))
    mean_q = np.zeros(num_nodes)
    norms_qp = np.zeros((len(pred_test), num_nodes))
    mean_qp = np.zeros(num_nodes)
    for i in range(len(pred_test)):
        for j in range(num_nodes):
            norms_q[i, j] = np.linalg.norm(pred_test[i, 4*j:4*j+2] - true_test[i,4*j:4*j+2])
            mean_q[j] = np.mean(norms_q[:, j])
            norms_qp[i, j] = np.linalg.norm(pred_test[i, 4*j+2:4*j+4] - true_test[i,4*j+2:4*j+4])
            mean_qp[j] = np.mean(norms_qp[:, j])
                
    if datacase == 1:
        fig1 = plt.figure(figsize = ((20, 15)))
        for i in range(1):
            plt.plot(true_test[i, np.arange(0, d, 4)], true_test[i, np.arange(1, d, 4)], '-', linewidth = 3, color = 'k', label = 'True')
            plt.plot(pred_test[i, np.arange(0, d, 4)], pred_test[i, np.arange(1, d, 4)], '--d', markersize = 5, linewidth = 1.8, color = 'r', label = 'Predicted')
        for i in np.arange(1,c,11): #remove ",11" from np.arange if you want to plot all test trajectories
            plt.plot(true_test[i, np.arange(0, d, 4)], true_test[i, np.arange(1, d, 4)], '-', linewidth = 3, color = 'k')
            plt.plot(pred_test[i, np.arange(0, d, 4)], pred_test[i, np.arange(1, d, 4)], '--d', markersize = 5, linewidth = 1.8, color = 'r')
        plt.xlabel(r"$q_x$", fontsize = "45")
        plt.ylabel(r"$q_y$", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.legend(fontsize = "45", loc = 'best')
        plt.title(r"Comparison over test trajectories $(q_x, q_y)$", fontsize = "45")
        plt.show()

        fig2 = plt.figure(figsize = ((20, 15)))
        circle = plt.Circle((0, 0), 1, color='k', alpha = 0.5, fill=False)
        plt.gca().add_patch(circle)
        for i in range(1):
            plt.scatter(true_test[i, np.arange(2, d, 4)], true_test[i, np.arange(3, d, 4)], color = 'k', s = 90, label = 'True')
            plt.scatter(pred_test[i, np.arange(2, d, 4)], pred_test[i, np.arange(3, d, 4)], color = 'r', s = 30, label = 'Predicted')
        for i in np.arange(1,c,11): #remove ",11" from np.arange if you want to plot all test trajectories
            plt.scatter(true_test[i, np.arange(2, d, 4)], true_test[i, np.arange(3, d, 4)], color = 'k', s = 90)
            plt.scatter(pred_test[i, np.arange(2, d, 4)], pred_test[i, np.arange(3, d, 4)], color = 'r', s = 30)
        
        plt.xlabel(r"$q^{\prime}_x$", fontsize = "45")
        plt.ylabel(r"$q^{\prime}_y$", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.axis('equal')
        plt.title(r"Comparison over test trajectories $(q^{\prime}_x, q^{\prime}_y)$", fontsize = "45")
        plt.legend(fontsize = "45", loc = 'center')
        plt.show()

        fig3 = plt.figure(figsize = ((20, 15)))
        plt.plot(np.linspace(0, 50, 51), mean_q, '-d', linewidth = 2, color = 'k', label = r"Error on $(q_x, q_y)$ ")
        plt.plot(np.linspace(0, 50, 51), mean_qp, '-d', linewidth = 2, color = 'r', label = r"Error on $(q^{\prime}_x, q^{\prime}_y)$")
        plt.xlabel(r"node $k$", fontsize = "45")
        plt.ylabel(r"Average norm of error", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.title(r"Mean error over test trajectories", fontsize = "45")
        plt.legend(fontsize = "45", loc = 'best')
        plt.show()

    
    if datacase == 2:
        fig1 = plt.figure(figsize = ((20, 15)))
        for i in range(1):
            plt.plot(true_test[i, np.arange(0, d, 4)], true_test[i, np.arange(1, d, 4)], '-', linewidth = 3, color = 'k', label = 'True')
            plt.plot(pred_test[i, np.arange(0, d, 4)], pred_test[i, np.arange(1, d, 4)], '--d', markersize = 5, linewidth = 1.8, color = 'r', label = 'Predicted')
        for i in np.arange(2,c,22): #remove ",22" from np.arange if you want to plot all test trajectories
            plt.plot(true_test[i, np.arange(0, d, 4)], true_test[i, np.arange(1, d, 4)], '-', linewidth = 3, color = 'k')
            plt.plot(pred_test[i, np.arange(0, d, 4)], pred_test[i, np.arange(1, d, 4)], '--d', markersize = 5, linewidth = 1.8, color = 'r')
        plt.xlabel(r"$q_x$", fontsize = "45")
        plt.ylabel(r"$q_y$", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.legend(fontsize = "45", loc = 'best')
        plt.title(r"Comparison over test trajectories $(q_x, q_y)$", fontsize = "45")
        plt.show()

        fig2 = plt.figure(figsize = ((20, 15)))
        circle = plt.Circle((0, 0), 1, color='k', alpha = 0.5, fill=False)
        plt.gca().add_patch(circle)
        for i in range(1):
            plt.scatter(true_test[i, np.arange(2, d, 4)], true_test[i, np.arange(3, d, 4)], color = 'k', s = 90, label = 'True')
            plt.scatter(pred_test[i, np.arange(2, d, 4)], pred_test[i, np.arange(3, d, 4)], color = 'r', s = 30, label = 'Predicted')
        for i in np.arange(2,c,22): #remove ",22" from np.arange if you want to plot all test trajectories
            plt.scatter(true_test[i, np.arange(2, d, 4)], true_test[i, np.arange(3, d, 4)], color = 'k', s = 90)
            plt.scatter(pred_test[i, np.arange(2, d, 4)], pred_test[i, np.arange(3, d, 4)], color = 'r', s = 30)
        
        
        
        plt.xlabel(r"$q^{\prime}_x$", fontsize = "45")
        plt.ylabel(r"$q^{\prime}_y$", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.axis('equal')
        plt.title(r"Comparison over test trajectories $(q^{\prime}_x, q^{\prime}_y)$", fontsize = "45")
        plt.legend(fontsize = "45", loc = 'center')
        plt.show()

        fig3 = plt.figure(figsize = ((20, 15)))
        plt.plot(np.linspace(0, 50, 51), mean_q, '-d', linewidth = 2, color = 'k', label = r"Error on $(q_x, q_y)$ ")
        plt.plot(np.linspace(0, 50, 51), mean_qp, '-d', linewidth = 2, color = 'r', label = r"Error on $(q^{\prime}_x, q^{\prime}_y)$")
        plt.xlabel(r"node $k$", fontsize = "45")
        plt.ylabel(r"Average norm of error", fontsize = "45")
        plt.tick_params(labelsize = "45")
        plt.title(r"Mean error over test trajectories", fontsize = "45")
        plt.legend(fontsize = "45", loc = 'best')
        plt.show()