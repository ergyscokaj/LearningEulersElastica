import torch
import numpy as np
from Scripts.utils import reconstruct_q_torch
from torch.func import jacfwd,vmap
import torch.nn as nn
import random


def trainModel(number_elements,device,model,criterion,optimizer,epochs,trainloader,valloader):

  torch.manual_seed(1)
  np.random.seed(1)

  L = 3.3
  lossVal = 1.
  epoch = 1

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)

  adapt_weights = torch.tensor([1,1])

  loss_constraint = 0.

  while epoch < epochs:
      losses = []
      running_loss = 0

      for i, inp in enumerate(trainloader):

          q1,q2,v1,v2,s,_,qs,vs = inp
          q1,q2,v1,v2,s,qs,vs = q1.to(device), q2.to(device), v1.to(device), v2.to(device), s.to(device), qs.to(device), vs.to(device)
          #PINN case

          s = s*L

          optimizer.zero_grad()
          loss = criterion(model(s,q1,q2,v1,v2),vs)

          #q comparison
          q_pred = torch.zeros_like(qs)
          q_pred[:,0] = reconstruct_q_torch(model,s,q1,q2,v1,v2,comp=0,k=10)
          q_pred[:,1] = reconstruct_q_torch(model,s,q1,q2,v1,v2,comp=1,k=10)
          loss_q = criterion(q_pred,qs)
          loss += loss_q

          loss.backward()
          optimizer.step()

      if epoch>0 and epoch%10==0:
        print(f'Loss [{epoch+1}](epoch): ', loss.item())

      if epoch==30 and loss.item()>1e-2:
        epoch = epochs+5
        loss = torch.tensor([torch.nan])
        print("Exited for too slow progress")
        break

      epoch += 1

      scheduler.step()

  print('Training Done')
  print("Loss : ",loss.item())
  return loss
