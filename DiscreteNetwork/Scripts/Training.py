import torch
import numpy as np
import random

def train(model,gamma,criterion,scheduler,optimizer,epochs,trainloader,testloader,device):
    
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    
    for epoch in range(epochs):
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
            
            loss.backward()
            optimizer.step()
        model.eval();
        with torch.no_grad():
            data = next(iter(testloader))
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            predicted = model(inputs)  
            
            predicted = torch.cat((inputs[:,:4],predicted,inputs[:,4:]),dim=1)
            labels = torch.cat((inputs[:,:4],labels,inputs[:,4:]),dim=1)
            test_error = criterion(predicted, labels)
            print(f'Loss [{epoch+1}](epoch): ', loss.item())
        model.train();
        scheduler.step()
    print('Training Done')
    print("Loss : ",loss.item())

    return loss