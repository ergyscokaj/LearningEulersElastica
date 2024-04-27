from torch.func import jacfwd,vmap
import torch
import torch.nn as nn
import numpy as np

# def trainModel(number_elements,device,model,criterion,optimizer,epochs,trainloader,train_with_tangents=False,pde_regularisation=True,soft_bcs_imposition=False):
  
#   torch.manual_seed(1)
#   np.random.seed(1)
  
#   lossVal = 1.
#   epoch = 1
#   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
#   x_eval = np.linspace(0,1,number_elements+1)
    
#   stored_res = 0
#   count = 0
#   cc = 0
#   is_good_loss = False
  
#   while epoch < epochs:
#       losses = []
#       running_loss = 0

#       for i, inp in enumerate(trainloader):
#           q1,q2,v1,v2,s,_,qs,vs = inp
#           q1,q2,v1,v2,s,qs,vs = q1.to(device),q2.to(device),v1.to(device),v2.to(device),s.to(device),qs.to(device),vs.to(device)

#           def closure():
#               optimizer.zero_grad()
#               res_q = model(s,q1,q2,v1,v2)
#               loss = criterion(res_q,qs) + criterion(model.derivative(s,q1,q2,v1,v2),vs) #Comparison only of the qs
#               loss += 1e-2 * torch.mean((torch.linalg.norm(model.derivative(s, q1, q2, v1, v2), ord=2, dim=1)**2-1.)**2)           
#               loss.backward()
#               return loss
            
#           optimizer.step(closure)
#       model.eval();
    
#       with torch.no_grad():
#         res_q = model(s,q1,q2,v1,v2)
#         loss = criterion(res_q,qs)
#         is_good_loss = (loss.item()<8e-6)
#         if epoch == 1:
#           stored_res = loss.item()
#         if epoch == 30:
#           check = (loss.item()>(stored_res * 1e-1))
#           if check and stored_res>1e-2:
#             print("Early stop due to lack of progress")
#             loss = torch.tensor(torch.nan)
#             epoch = epochs + 10
#         print(f'Loss [{epoch+1}](epoch): ', loss.item())
#         if torch.isnan(loss):
#           epoch = epochs + 10
#           break
            
#       model.train();
#       epoch += 1
#       scheduler.step()

#   print('Training Done')
#   print("Loss : ",loss.item())
#   return loss