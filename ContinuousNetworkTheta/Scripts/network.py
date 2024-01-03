import torch
import torch.nn as nn
from torch.func import jacfwd, vmap
import numpy as np

from numpy.core.multiarray import vdot


class act(nn.Module):
  def __init__(self,):
    super().__init__()
  def forward(self,x):
    return torch.sigmoid(x)*x

class theta_net(nn.Module):
  def __init__(self,act_name,nlayers,hidden_nodes,is_res,is_deeponet):
    super().__init__()

    torch.manual_seed(1)
    np.random.seed(1)

    if act_name=='tanh':
      self.act = lambda x : torch.tanh(x)
    elif act_name=="sin":
      self.act = lambda x : torch.sin(x)
    elif act_name=="swish":
      self.act = lambda x: x*torch.sigmoid(x)
    else:
      self.act = lambda x : torch.sigmoid(x)

    self.hidden_nodes = hidden_nodes
    self.nlayers = nlayers
    self.is_res = is_res
    self.is_deeponet = is_deeponet

    self.embed = nn.Linear(3,self.hidden_nodes)

    # the next foru lines are layers of DeepONet trunk
    self.lift_U = nn.Linear(self.hidden_nodes,self.hidden_nodes)
    self.lift_V = nn.Linear(self.hidden_nodes,self.hidden_nodes)

    self.lift_H = nn.Linear(self.hidden_nodes,self.hidden_nodes) #added

    self.linears_Z = nn.ModuleList([nn.Linear(self.hidden_nodes,self.hidden_nodes) for i in range(self.nlayers)])

    self.lift = nn.Linear(self.hidden_nodes,self.hidden_nodes)

    self.linears = nn.ModuleList([nn.Linear(self.hidden_nodes,self.hidden_nodes) for _ in range(self.nlayers)])
    self.linearsO = nn.ModuleList([nn.Linear(self.hidden_nodes,self.hidden_nodes) for _ in range(self.nlayers)])

    self.proj = nn.Linear(self.hidden_nodes,1)

  def find_theta(self,v):
    return torch.atan2(v[:,1:2], v[:,0:1])

  #This is N in the markdown cell above
  def forward(self,s,q1,q2,v1,v2):
    s = s.reshape(-1,1)# / L
    q1 = q1.reshape(-1,2)
    q2 = q2.reshape(-1,2)
    v1 = v1.reshape(-1,2)
    v2 = v2.reshape(-1,2)

    theta1 = self.find_theta(v1) / torch.pi
    theta2 = self.find_theta(v2) / torch.pi

    #input = torch.cat((theta1,theta2,q1,q2,s),dim=1)
    input = torch.cat((theta1,theta2,s),dim=1)
    input = torch.sin(2*torch.pi*self.embed(input))

    if self.is_deeponet:
        U = self.act(self.lift_U(input))
        V = self.act(self.lift_V(input))

        H = self.act(self.lift_H(input)) #added

        for i in range(self.nlayers):
            Z = self.linearsO[i](self.act(self.linears_Z[i](H)))
            H = U*(1-Z) + V*Z

        input = H

    else:

        input = self.act(self.lift(input))

        for i in range(self.nlayers):
          if self.is_res:
              input = input + self.linearsO[i](self.act(self.linears[i](input)))
          else:
              input = self.act(self.linears[i](input))

    output = self.proj(input)
    return output*torch.pi

class network(nn.Module):
        def __init__(self,L=3.3,impose_bcs=True,act_name='sin', nlayers=3, hidden_nodes = 100, is_res=True, is_deeponet=False):
          super().__init__()

          torch.manual_seed(1)
          np.random.seed(1)

          self.L = L
          self.impose_bcs = impose_bcs
          self.parametric_part = theta_net(act_name, nlayers, hidden_nodes, is_res, is_deeponet)

        def find_theta(self,v):
          return torch.atan2(v[:,1:2], v[:,0:1])

        def local_poly_left(self,s,d=1/30):
          return (d - s)**3/d**3  #polynomial which is 1 at s=0, and at s=d it vanishes together with its first two derivatives
        def local_poly_right(self,s,d=1/30):
          return (d + s - self.L)**3/d**3 #same but for the right node

        def theta(self,s,q1,q2,v1,v2):

          s = s.reshape(-1,1)
          q1 = q1.reshape(-1,2)
          q2 = q2.reshape(-1,2)
          v1 = v1.reshape(-1,2)
          v2 = v2.reshape(-1,2)

          parametric_part = self.parametric_part(s,q1,q2,v1,v2)

          if self.impose_bcs:
            left_node = torch.zeros_like(s).to(s.device)
            right_node = (torch.ones_like(s)*self.L).to(s.device)

            theta_1 = self.find_theta(v1)
            theta_2 = self.find_theta(v2)


            g_left = self.parametric_part(left_node,q1,q2,v1,v2).to(s.device)
            g_right =  self.parametric_part(right_node,q1,q2,v1,v2).to(s.device)

            return  parametric_part + (theta_1-g_left) * torch.exp(-100*s**2) + (theta_2-g_right) * torch.exp(-100*(s-self.L)**2)
          else:
            return parametric_part

        def theta_prime(self,s,q1,q2,v1,v2):
          s = s.reshape(-1,1)
          q1 = q1.reshape(-1,2)
          q2 = q2.reshape(-1,2)
          v1 = v1.reshape(-1,2)
          v2 = v2.reshape(-1,2)
          res = vmap(jacfwd(self.theta,argnums=0))(s,q1,q2,v1,v2)
          return res[:,0,:,0]

        def theta_second(self,s,q1,q2,v1,v2):
          s = s.reshape(-1,1)
          q1 = q1.reshape(-1,2)
          q2 = q2.reshape(-1,2)
          v1 = v1.reshape(-1,2)
          v2 = v2.reshape(-1,2)
          res = vmap(jacfwd(self.theta_prime,argnums=0))(s,q1,q2,v1,v2)
          return res[:,0,:,0]

        def theta_third(self,s,q1,q2,v1,v2):
          s = s.reshape(-1,1)
          q1 = q1.reshape(-1,2)
          q2 = q2.reshape(-1,2)
          v1 = v1.reshape(-1,2)
          v2 = v2.reshape(-1,2)
          return vmap(jacfwd(self.theta_second,argnums=0))(s,q1,q2,v1,v2)[:,0,:,0]

        def forward(self,s,q1,q2,v1,v2):
          theta = self.theta(s,q1,q2,v1,v2)
          return torch.cat((torch.cos(theta),torch.sin(theta)),dim=1)