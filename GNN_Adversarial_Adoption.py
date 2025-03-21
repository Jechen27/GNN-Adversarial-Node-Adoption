import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph import utils
import sys
from deeprobust.graph.targeted_attack import Nettack
from MRPFramework import AdversarialNetwork
from MRPFramework import MRP
from NewAttacks import GraphSAGE, replaceNettackLabel, TargetedMetaAttack, TargetedPGDAttack
from Priority import PriorityApproach

#Usage: python file dataset_name GNN_name agent_name

dataset_name = sys.argv[1]
GNN_name = sys.argv[2]
agent_name = sys.argv[3]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = Dataset(root='/tmp/', name=dataset_name)

adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

adj, features, labels = utils.preprocess(adj, features, labels)

class_counts = np.bincount(labels)

print("#Nodes in each class")
print(class_counts)

classidx1 = int(input("input "+ agent_name + " team label: "))
classidx2 = int(input("input Priority team label: "))

team1_class_nodes = np.where(labels == classidx1)[0]
team2_class_nodes = np.where(labels == classidx2)[0]

other_class_nodes = np.where((labels != classidx1) & (labels != classidx2))[0]

print("Number of nodes in the team1 class:", len(team1_class_nodes))
print("Number of nodes in the team2 class:", len(team2_class_nodes))
print("Number of nodes in the other classes:", len(other_class_nodes))

if(GNN_name.lower() == 'gcn'):
    surrogate_model = GCN(nfeat=features.shape[1], nhid=16, nclass=len(torch.unique(labels)), device=device)
elif(GNN_name.lower() == 'sage'):
    surrogate_model = GraphSAGE(in_channels=features.shape[1], hidden_channels=16, out_channels=len(torch.unique(labels)))

default_model = GCN(nfeat=features.shape[1], nhid=16, nclass=len(torch.unique(labels)), device=device)
surrogate_model = surrogate_model.to(device)
optimizer = torch.optim.Adam(surrogate_model.parameters(), lr=0.01, weight_decay=5e-4)

def train_surrogate(model, features, adj, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = torch.nn.functional.nll_loss(output[idx_train], labels[idx_train])
    loss.backward()
    optimizer.step()
    return loss.item()

def predict_surrogate(model, data):
    model.eval()
    return model(data.x, data.edge_index)

for epoch in range(200):
    loss = train_surrogate(surrogate_model, features, adj, labels, data.idx_train)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss}")

replaceNettackLabel(classidx1)

if agent_name.lower() == 'mrp':
    agent1 = MRP(surrogate_model, features, classidx1, adj.shape[0], other_class_nodes)
elif agent_name.lower() == 'nettack':
    agent1 = Nettack(default_model, nnodes=adj.shape[0], attack_structure=True, attack_features=False, device=device).to(device)
elif agent_name.lower() == 'metattack':
    agent1 = TargetedMetaAttack(model=default_model, nnodes=adj.shape[0], target_label=classidx1, device=device).to(device)
elif agent_name.lower() == 'pgd':
    agent1 = TargetedPGDAttack(model=default_model, nnodes=adj.shape[0], target_label=classidx1, device=device).to(device)


agent2 = PriorityApproach(model=surrogate_model,target_class=classidx2)

game = AdversarialNetwork(adj, features, labels, surrogate_model, agent1, classidx1, agent2, classidx2, 10, 100, device)

for i in range(10):
    game.runRound()

team1_class_nodes_after = np.where(game.labels == classidx1)[0]
team2_class_nodes_after = np.where(game.labels == classidx2)[0]

print('team 1 nodes converted: ', len(team1_class_nodes_after) - len(team1_class_nodes))
print('team 2 nodes converted: ', len(team2_class_nodes_after) - len(team2_class_nodes))