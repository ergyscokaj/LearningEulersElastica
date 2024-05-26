from torch.func import jacfwd,vmap
import torch
import torch.nn as nn
import numpy as np

def trainModel(number_elements,device,model,criterion,optimizer,epochs,trainloader,valloader,train_with_tangents=False,pde_regularisation=True,soft_bcs_imposition=False):
  
    torch.manual_seed(1)
    np.random.seed(1)

    lossVal = 1.
#     epoch = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
    x_eval = np.linspace(0,1,number_elements+1)

    stored_res = 0
    count = 0
    cc = 0
    is_good_loss = False


    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        losses = []
        running_loss = 0

        for i, inp in enumerate(trainloader):
            q1,q2,v1,v2,s,_,qs,vs = inp
            q1,q2,v1,v2,s,qs,vs = q1.to(device),q2.to(device),v1.to(device),v2.to(device),s.to(device),qs.to(device),vs.to(device)


            optimizer.zero_grad()
            res_q = model(s,q1,q2,v1,v2)
            loss = criterion(res_q,qs) + criterion(model.derivative(s,q1,q2,v1,v2),vs) #Comparison only of the qs
            loss += 1e-2 * torch.mean((torch.linalg.norm(model.derivative(s, q1, q2, v1, v2), ord=2, dim=1)**2-1.)**2)           
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
        
        # Calculate average training loss
        train_loss_avg = running_loss / len(trainloader)
        train_losses.append(train_loss_avg)
        
        model.eval();
        
        val_loss = 0
        with torch.no_grad():
            for inp in valloader:
                q1_val, q2_val, v1_val, v2_val, s_val, _, qs_val, vs_val = inp
                q1_val, q2_val, v1_val, v2_val, s_val, qs_val, vs_val = q1_val.to(device), q2_val.to(device), v1_val.to(device), v2_val.to(device), s_val.to(device), qs_val.to(device), vs_val.to(device)

                res_q_val = model(s_val, q1_val, q2_val, v1_val, v2_val)
                val_loss += criterion(res_q_val, qs_val).item()

            # Calculate average validation loss
            val_loss_avg = val_loss / len(valloader)
            val_losses.append(val_loss_avg)           

            res_q = model(s,q1,q2,v1,v2)
            loss = criterion(res_q,qs)

            is_good_loss = (loss.item()<8e-6)
            if epoch == 1:
                stored_res = loss.item()
            if epoch == 30:
                check = (loss.item()>(stored_res * 1e-1))
                if check:
                    print("Early stop due to lack of progress")
                    loss = torch.tensor(torch.nan)
                    epoch = epochs + 10
            if torch.isnan(loss):
                epoch = epochs + 10
                break

        model.train();
        epoch += 1
        scheduler.step()

    print('Training Done')
    return loss