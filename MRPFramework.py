from deeprobust.graph.data import Dataset
from deeprobust.graph.targeted_attack import Nettack
import torch
import torch.nn.functional as F
import numpy as np
from NewAttacks import TargetedMetaAttack, TargetedPGDAttack
from Priority import perturb_node_to_target_class
import math
import random


class AdversarialNetwork:
    def __init__(self, adj, features, labels, gnn, agent1, idx1, agent2, idx2, rounds, budget, idx_train, idx_unlabeled, device):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.gnn = gnn
        self.agent1=agent1
        self.idx1=idx1
        self.agent2=agent2
        self.idx2=idx2
        self.rounds=rounds
        self.curRound = 0
        self.budget=budget
        self.budget1=budget
        self.budget2=budget
        self.idx_train=idx_train
        self.idx_unlabeled=idx_unlabeled
        self.device=device

    def runRound(self):
        if self.curRound >= self.rounds:
            print("All rounds completed.")
            return
        if isinstance(self.agent1, MRP):
            # Use Monte Carlo Tree Search (MCTS) to determine the budget for MRP
            round_budget1 = self.agent1.monte_carlo_tree_search(self.adj, self.features, self.labels, self.budget1, self.curRound, self.rounds)
        else:
            round_budget1 = self.budget // self.rounds
        
        round_budget2 = self.budget // self.rounds

        if self.budget1 > 0:
            used_budget1 = 0

            if isinstance(self.agent1, Nettack):  
                candidate_nodes = np.where(self.labels != self.idx1)[0]
                np.random.shuffle(candidate_nodes) 

                for node in candidate_nodes:
                    if used_budget1 >= round_budget1:
                        break 

                    # Attack the node with Nettack
                    while self.labels[node] != self.idx1 and used_budget1 < round_budget1:
                        self.agent1.attack(features=self.features, adj=self.adj, labels=self.labels, target_node=node, n_perturbations=1)
                        modified_adj1 = torch.tensor(self.agent1.modified_adj.toarray())  
                        used_budget1 += 1 

                        # Update labels after the attack
                        self.labels = self.gnn(self.features, self.adj).argmax(axis=1)
            elif isinstance(self.agent1, MRP):
                self.agent1.update(self.agent2.perturbations, self.features, self.adj, self.labels)
                modified_adj1 = self.agent1.attack(self.adj, self.features, self.labels, round_budget1)
                self.adj = modified_adj1
                used_budget1 = round_budget1
            elif isinstance(self.agent1, TargetedMetaAttack):
                self.agent1.attack(
                ori_adj=self.adj,
                ori_features=self.features,
                labels=self.labels,
                idx_train=self.idx_train,
                idx_unlabeled=self.idx_unlabeled,
                n_perturbations=round_budget1
            )
                modified_adj1 = self.agent1.modified_adj
            elif isinstance(self.agent1, TargetedPGDAttack):
                self.agent1.attack(
                ori_adj=self.adj,
                ori_features=self.features,
                labels=self.labels,
                idx_train=self.idx_train,
                n_perturbations=round_budget1
            )
                modified_adj1 = self.agent1.modified_adj
            else:
                # For other agents (e.g., PriorityApproach)
                modified_adj1 = self.agent1.attack(self.adj, self.features, self.labels, round_budget1, device=self.device)
                self.adj = modified_adj1
                used_budget1 = round_budget1 

        self.adj = modified_adj1
        self.labels = self.gnn(self.features, self.adj).argmax(axis=1)
        self.budget1 -= used_budget1  

        # Run agent2's attack
        if self.budget2 > 0:
            used_budget2 = min(round_budget2, self.budget2)  # Use remaining budget if less than round_budget
            modified_adj2 = self.agent2.attack(self.adj, self.features, self.labels, used_budget2, device=self.device)
            self.adj = modified_adj2 
            self.budget2 -= used_budget2 

            self.labels = self.gnn(self.features, self.adj).argmax(axis=1)

        self.curRound += 1

class MCTSNode:
    def __init__(self, adj, features, labels, remaining_budget, curRound, rounds, parent, value=0):
        #Initialize an MCTS node
        self.adj = adj
        self.features = features
        self.labels = labels
        self.remaining_budget = remaining_budget
        self.curRound = curRound
        self.rounds = rounds
        self.value = value  # Average reward from simulations
        self.visits = 0  # Number of times the node has been visited
        self.children = {}  # Maps actions to child nodes
        self.parent = parent

class MRP:
    def __init__(self, model, features, target_class, num_nodes, other_class_nodes, device='cpu'):
        self.model = model
        self.features = features
        self.target_class = target_class
        self.device = device
        self.num_nodes = num_nodes
        self.regret = torch.ones(num_nodes, device=device) / num_nodes
        self.other_class_nodes = other_class_nodes

    def attack(self, adj, features, labels, budget, **kwargs):
        selected_nodes = torch.multinomial(self.regret, budget, replacement=False)
        modified_adj = adj.clone() 
        remaining_budget = budget

        for node in selected_nodes:
            while remaining_budget > 0:
                modified_adj, remaining_budget = perturb_node_to_target_class(self.model, adj, features, labels, node, self.target_class, remaining_budget)

        self.regret /= self.regret.sum()

        return modified_adj
    
    def update(self, selected_nodes, features, adj, labels):
        selected_nodes = torch.tensor(selected_nodes, device=self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(features, adj)

        losses = F.nll_loss(output[selected_nodes], labels[selected_nodes], reduction='none')

        # Normalize losses
        losses = losses / losses.sum()

        discount = 0.8

        # Decrease the probability of selecting nodes that resulted in high loss
        self.regret[selected_nodes] *= discount*(1 - losses)

        # Ensure regret probabilities remain valid (non-negative and sum to 1)
        self.regret = torch.clamp(self.regret, min=0)  # Ensure non-negative probabilities
        self.regret /= self.regret.sum()  # Normalize

    def monte_carlo_tree_search(self, adj, features, labels, remaining_budget, curRound, rounds, num_simulations=100):
        root = MCTSNode(adj, features, labels, remaining_budget, curRound, rounds, None)

        for _ in range(num_simulations):
            # Step 1: Selection
            node = self.select(root)

            # Step 2: Expansion
            if node.curRound < node.rounds and node.remaining_budget > 0:
                self.expand(node)

            # Step 3: Simulation
            reward = self.simulate(node)

            # Step 4: Backpropagation
            self.backpropagate(node, reward)

        # Choose the action with the highest value
        best_action = max(root.children.keys(), key=lambda a: root.children[a].value)
        return int(best_action)

    def select(self, node):
        while node.children:
            # Use UCB1 to select the best child
            node = max(node.children.values(), key=lambda n: n.value + (2 * math.log(node.visits) / n.visits) ** 0.5)
        return node

    def expand(self, node):
        if node.remaining_budget <= 0 or node.curRound >= node.rounds:
            return  

        # Possible actions: Spend budget on a node or save budget
        for node_id in self.other_class_nodes:
            if node_id not in node.children:
                # Add child node for spending budget
                new_adj = self.perturb_node(node.adj, node_id)
                new_budget = node.remaining_budget - 1
                node.children[node_id] = MCTSNode(new_adj, node.features, node.labels, new_budget, node.curRound, node.rounds, node, 0.05)

        # Add child node for saving budget
        if "save" not in node.children:
            node.children["save"] = MCTSNode(node.adj, node.features, node.labels, node.remaining_budget, node.curRound + 1, node.rounds, node)

    def simulate(self, node):
        # Clone the current state
        sim_adj = node.adj.clone()
        sim_budget = node.remaining_budget
        sim_round = node.curRound

        while sim_budget > 0 and sim_round < node.rounds:
            spend_this_round = random.randint(1, min(sim_budget, 5))

            # Perturb nodes using the allocated budget
            for _ in range(spend_this_round):
                node_to_attack = random.choice(self.other_class_nodes)
                sim_adj = self.perturb_node(sim_adj, node_to_attack)

            sim_budget -= spend_this_round
            sim_round += 1 

        # Evaluate the final attack success
        with torch.no_grad():
            output = self.model(node.features, sim_adj)
            target_nodes_mask = (node.labels != self.target_class)
            success_rate = (output[target_nodes_mask].argmax(1) == self.target_class).float().mean()

        return success_rate.item() 

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += (reward - node.value) / node.visits
            node = node.parent 

    def perturb_node(self, adj, node_id):
        if isinstance(adj, torch.Tensor):
            modified_adj = adj
        else:
            modified_adj = torch.tensor(adj.toarray())
        num_nodes = modified_adj.size(0)
        all_edges = torch.LongTensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]).t()

        edge_weight_dense = modified_adj.clone().detach().requires_grad_(True)

        output = self.model(self.features, all_edges, edge_weight_dense[all_edges[0], all_edges[1]])

        loss = F.nll_loss(output[[node_id]], torch.tensor([self.target_class], device=output.device))
        print("Loss:", loss.item())
        
        self.model.zero_grad()
        loss.backward()

        # Get gradients with respect to the edge weights
        edge_weight_grad = edge_weight_dense.grad

        # Find the edge with the maximum gradient (excluding self-loops and existing edges)
        edge_weight_grad[node_id, node_id] = -float('inf')  # Ignore self-loops
        edge_weight_grad[modified_adj == 1] = -float('inf')  # Ignore existing edges

        best_grad = torch.argmax(edge_weight_grad).item()
        u, v = torch.div(best_grad, num_nodes, rounding_mode='floor').item(), (best_grad % num_nodes).item()
        if (modified_adj[u, v] != 1):
            modified_adj[u, v] = 1
            modified_adj[v, u] = 1

        # Reset gradients
        modified_adj.grad = None

        return modified_adj.detach()
