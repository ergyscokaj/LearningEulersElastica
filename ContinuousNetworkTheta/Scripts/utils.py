import numpy as np
import torch

#It extracts the entries correspondent to boundary conditions

def getBCs(trajectories):
    bcs = {"q1":trajectories[:,:2],
       "q2":trajectories[:,-4:-2],
       "v1":trajectories[:,2:4],
       "v2":trajectories[:,-2:]}
    return bcs
 
def reconstruct_q_comp(q1,q2,v1,v2,model,upper,comp=0,k=10):
   #From left to right
   integrand = lambda s,q1,q2,v1,v2 : model(s,q1,q2,v1,v2)[:,comp:comp+1]

   bs = len(v1)
   tt = torch.linspace(0,upper,k+1).unsqueeze(1).to(q1.device)
   ba = tt[1]-tt[0]

   w1 = ba * 8/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
   w2 = ba * 5/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
   w3 = ba * 5/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
   x1 = (tt[:-1]+tt[1:])/2  #mapping of 0 node
   x2 = (tt[:-1]+tt[1:])/2  + 0.5 * (-np.sqrt(3/5)) * ba #mapping of -sqrt(3/5) node
   x3 = (tt[:-1]+tt[1:])/2  + 0.5 * (np.sqrt(3/5)) * ba #mapping of sqrt(3/5) node

   quad_nodes = torch.cat((x1,x2,x3),dim=0).to(q1.device)
   quad_weights = torch.cat((w1,w2,w3),dim=0).to(q1.device)
   ones_like_nodes = torch.ones_like(quad_nodes).to(q1.device)
   ones_like_bcs = torch.ones((len(v1),1)).to(q1.device)

   s = torch.kron(ones_like_bcs,quad_nodes)
   w = torch.kron(ones_like_bcs,quad_weights)
   q1_kron = torch.kron(q1,ones_like_nodes)
   q2_kron = torch.kron(q2,ones_like_nodes)
   v1_kron = torch.kron(v1,ones_like_nodes)
   v2_kron = torch.kron(v2,ones_like_nodes)


   q_upper_comp_left = q1[:,comp] + torch.sum((integrand(s,q1_kron,q2_kron,v1_kron,v2_kron)*w).reshape(bs,3*k),dim=1)
   return q_upper_comp_left

num_elements = 50

def reconstruct_q(q1,q2,v1,v2,L,model,device):

  beam_nodes = np.linspace(0,L,num_elements+1)

  q1t,q2t,v1t,v2t = torch.from_numpy(q1.astype(np.float32)).to(device), torch.from_numpy(q2.astype(np.float32)).to(device), torch.from_numpy(v1.astype(np.float32)).to(device), torch.from_numpy(v2.astype(np.float32)).to(device)

  q = np.zeros((len(q1),2,(num_elements+1)))

  for count,upper in enumerate(beam_nodes):
    q[:,0,count] = reconstruct_q_comp(q1t,q2t,v1t,v2t,model,upper,comp=0,k=10).detach().cpu().numpy()
    q[:,1,count] = reconstruct_q_comp(q1t,q2t,v1t,v2t,model,upper,comp=1,k=10).detach().cpu().numpy()

  return q

def reconstruct_q_torch(model,node,q1,q2,v1,v2,comp=0,k=10):


  #From left to right
  integrand = lambda s,q1,q2,v1,v2 : model(s,q1,q2,v1,v2)[:,comp:comp+1]

  # Expand the tensor s to match the desired output shape
  expanded_s = node.expand(-1, k+1)
  # Create a range tensor from 0 to s[i] for each row
  range_tensor = torch.linspace(0, 1, k+1).to(q1.device)
  # Multiply the range tensor with the expanded_s element-wise
  tt = (expanded_s * range_tensor).to(q1.device)

  ba = node/k #different for each row

  bs = len(v1)

  w1 = ba * 8/9 / 2
  w2 = ba * 5/9 / 2
  w3 = ba * 5/9 / 2

  weights = torch.cat((w1.repeat(1,k),w2.repeat(1,k),w2.repeat(1,k)),dim=1).to(q1.device) #dim bs x 3k

  #Nodes stored in tensor of size bs x k
  x1 = (tt[:,:-1]+tt[:,1:])/2  #mapping of 0 node
  x2 = (tt[:,:-1]+tt[:,1:])/2  + 0.5 * (-np.sqrt(3/5)) * ba #mapping of -sqrt(3/5) node
  x3 = (tt[:,:-1]+tt[:,1:])/2  + 0.5 * (np.sqrt(3/5)) * ba #mapping of sqrt(3/5) node

  nodes = torch.cat((x1,x2,x3),dim=1).to(q1.device) #dim bs x 3k
  nodes_flat = nodes.reshape(-1,1)

  ones_like_nodes = torch.ones(int(3*k),1).to(q1.device)

  q1_kron = torch.kron(q1,ones_like_nodes)
  q2_kron = torch.kron(q2,ones_like_nodes)
  v1_kron = torch.kron(v1,ones_like_nodes)
  v2_kron = torch.kron(v2,ones_like_nodes)

  result = integrand(nodes_flat,q1_kron,q2_kron,v1_kron,v2_kron)*weights.reshape(-1,1)

  q_upper_comp_left = q1[:,comp] + torch.sum(result.reshape(bs,3*k),dim=1)

  return q_upper_comp_left.view(-1)

def compatibility_condition(q1,q2,v1,v2,model,comp=0,k=5):
  #Gaussian quadrature on k nodes
  #comp determines if we integrate the x or y component, respectively with comp=0 and 1.

  integrand = lambda s,q1,q2,v1,v2 : model(s,q1,q2,v1,v2)[:,comp:comp+1]

  #q1_og = q1.detach().clone()
  #q2_og = q2.detach().clone()

  bs = len(v1)
  tt = torch.linspace(0,L,k+1).unsqueeze(1).to(q1.device)
  ba = tt[1]-tt[0] #divide the interval [0,1] into k subintervals of size ba, over which we apply Gaussian 3 point quadrature


  w1 = ba * 8/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
  w2 = ba * 5/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
  w3 = ba * 5/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
  x1 = (tt[:-1]+tt[1:])/2  #mapping of 0 node
  x2 = (tt[:-1]+tt[1:])/2  + 0.5 * (-np.sqrt(3/5)) * ba #mapping of -sqrt(3/5) node
  x3 = (tt[:-1]+tt[1:])/2  + 0.5 * (np.sqrt(3/5)) * ba #mapping of sqrt(3/5) node

  quad_nodes = torch.cat((x1,x2,x3),dim=0).to(q1.device)
  quad_weights = torch.cat((w1,w2,w3),dim=0).to(q1.device)
  ones_like_nodes = torch.ones_like(quad_nodes).to(q1.device)
  ones_like_bcs = torch.ones((len(v1),1)).to(q1.device)

  s = torch.kron(ones_like_bcs,quad_nodes)
  w = torch.kron(ones_like_bcs,quad_weights)
  q1_kron = torch.kron(q1,ones_like_nodes)
  q2_kron = torch.kron(q2,ones_like_nodes)
  v1_kron = torch.kron(v1,ones_like_nodes)
  v2_kron = torch.kron(v2,ones_like_nodes)


  compatibility_comp = torch.mean((torch.sum((integrand(s,q1_kron,q2_kron,v1_kron,v2_kron)*w).reshape(bs,3*k),dim=1) - (q2-q1)[:,comp])**2  )

  return compatibility_comp