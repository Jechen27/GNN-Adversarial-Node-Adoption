import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import sys
from deeprobust.graph.global_attack import BaseAttack
from deeprobust.graph import utils

def perturb_node_to_target_class(model, adj, features, labels, target_node, target_class, budget):
    if isinstance(adj, torch.Tensor):
        modified_adj = adj
    else:
        modified_adj = torch.tensor(adj.toarray())
    remaining_budget = budget
    perturbation_count=0
    num_nodes = modified_adj.size(0)
    all_edges = torch.LongTensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]).t()

    edge_weight_dense = modified_adj.clone().detach().requires_grad_(True)

    while remaining_budget > 0:
        output = model(features, all_edges, edge_weight_dense[all_edges[0], all_edges[1]])
        predicted_class = torch.argmax(output[target_node]).item()

        # Check if the node has been converted
        if predicted_class == target_class:
            print(f"Node {target_node} has been converted to class {target_class} after {perturbation_count}.")
            break

        loss = F.nll_loss(output[[target_node]], torch.tensor([target_class], device=output.device))
        print("Loss:", loss.item())
        
        model.zero_grad()
        loss.backward()

        # Get gradients with respect to the edge weights
        edge_weight_grad = edge_weight_dense.grad

        # Find the edge with the maximum gradient
        edge_weight_grad[target_node, target_node] = -float('inf')  # Ignore self-loops
        edge_weight_grad[modified_adj == 1] = -float('inf')  # Ignore existing edges

        sorted_grads = edge_weight_grad.view(-1).argsort(descending=False)
        for max_grad_index in sorted_grads:
            u, v = torch.div(max_grad_index, num_nodes, rounding_mode='floor').item(), (max_grad_index % num_nodes).item()
            if (modified_adj[u, v] != 1):
                modified_adj[u, v] = 1
                modified_adj[v, u] = 1
                remaining_budget -= 1
                perturbation_count += 1
                print(f"Added edge between {u} and {v}. Remaining budget: {remaining_budget}")
                break

        modified_adj.grad = None

    return modified_adj.detach(), remaining_budget


class PriorityApproach(BaseAttack):
    def __init__(self, model=None, target_class=None, device='cpu'):
        super(PriorityApproach, self).__init__(model, device)
        self.model=model
        self.target_class = target_class
        self.device = device
        self.perturbations = set()

    def attack(self, adj, features, labels, budget, **kwargs):
        self.perturbations.clear()

        adj, features, labels = utils.to_tensor(adj, features, labels)

        high_degree_nodes = self.get_high_degree_nodes(adj, labels, budget)

        perturbation_count = 0

        modified_adj = sp.csr_matrix(adj.numpy()).tolil()

        remaining_budget = budget

        for node in high_degree_nodes:
            while remaining_budget > 0:
                # Find a node in the target class to connect to
                node2 = np.random.choice(np.where(labels == self.target_class)[0])

                # Add the edge if it doesn't already exist
                if modified_adj[node, node2] == 0:
                    modified_adj[node, node2] = 1
                    modified_adj[node2, node] = 1
                    perturbation_count += 1
                    remaining_budget -= 1
                    self.perturbations.add(node)
                    self.perturbations.add(node2)
                    print(f"Added edge between {node} and {node2}")
                    predicted_labels = self.model(features, torch.tensor(modified_adj.toarray())).argmax(axis=1)
                    if predicted_labels[node] == self.target_class:
                        print(f"{node} has been converted to {self.target_class} after {perturbation_count} perturbations")
                        break
                # alternative edge addition below
                # modified_adj, remaining_budget = perturb_node_to_target_class(self.model, modified_adj, features, labels, node, self.target_class, remaining_budget)
        return torch.tensor(modified_adj.toarray())

    def get_high_degree_nodes(self, adj, labels, budget):
        degrees = np.array(adj.sum(1)).flatten()

        high_degree_nodes = [node for node in range(adj.shape[0]) if labels[node] != self.target_class]

        high_degree_nodes = sorted(high_degree_nodes, key=lambda x: degrees[x], reverse=True)

        high_degree_nodes = high_degree_nodes[:budget]

        return high_degree_nodes
    
