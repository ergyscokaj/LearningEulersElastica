import torch
import numpy as np
import random
import matplotlib.pyplot as plt

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model,gamma,criterion,scheduler,optimizer,epochs,trainloader,valloader,device):
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    early_stopper = EarlyStopper(patience=100, min_delta=0)
    losses = []
    losses_val = []
    for epoch in range(epochs):
        
        train_loss = 0.
        counter = 0.
        
        for _, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            predicted = model(inputs) 

            loss = criterion(predicted, labels) 
            
            predicted = torch.cat((inputs[:,:4],predicted,inputs[:,4:]),dim=1)
            labels = torch.cat((inputs[:,:4],labels,inputs[:,4:]),dim=1)
            
            predicted_first = predicted[:,:-4]
            predicted_forward = predicted[:,4:]
            
            labels_first = labels[:,:-4]
            labels_forward = labels[:,4:]
            
            diff_predicted = predicted_forward - predicted_first
            diff_labels = labels_forward - labels_first
            
            loss += gamma * criterion(diff_predicted, diff_labels)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            counter+=1
        
        avg_train_loss = train_loss/counter
        losses.append(avg_train_loss)
        
        model.eval();
        with torch.no_grad():
            data = next(iter(valloader))
            inputs, labels = data[0].to(device), data[1].to(device)
            predicted = model(inputs)  
            #predicted = torch.cat((inputs[:,:4],predicted,inputs[:,4:]),dim=1)
            #labels = torch.cat((inputs[:,:4],labels,inputs[:,4:]),dim=1)
            val_loss = criterion(predicted, labels)    
        model.train();
        losses_val.append(val_loss.item())
        
        if epoch%int(0.1*epochs)==0 and epoch>1:
            print(f"The average loss in epoch {epoch+1} is ",avg_train_loss)

        '''if early_stopper.early_stop(val_loss):  
            print("Early stopping")    
            #epochs = epoch + 1     
            break'''
        
        #scheduler.step(val_loss)
        scheduler.step()
    print('Training Done')
    
    '''plt.semilogy(np.arange(epochs),losses,'r-*',label="training loss",markersize=1)
    plt.semilogy(np.arange(epochs),losses_val,'k-d',label="validation loss",markersize=1)
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.legend()
    plt.title("Training and validation losses")
    plt.show();'''

    return loss