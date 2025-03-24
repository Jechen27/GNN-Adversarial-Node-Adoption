import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GraphConv
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.global_attack import Metattack
from deeprobust.graph.global_attack import PGDAttack
from deeprobust.graph import utils

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super(GraphSAGE, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))
        self.convs.append(GraphConv(hidden_channels, out_channels))
        self.nfeat = in_channels
        self.hidden_sizes = [hidden_channels]
        self.nclass = out_channels

    def forward(self, x, adj, edge_weight=None):
        edge_index = adj.nonzero().t().contiguous()  # Convert dense adj to sparse edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight=edge_weight)  # Use edge_index for SAGEConv
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)
    
def replaceNettackLabel(label):
    def newF(self, logits):
        return label
    Nettack.strongest_wrong_class = newF

class TargetedMetaAttack(Metattack):
    #Modified Metattack to perturb nodes toward a target label.

    def __init__(self, target_label, **kwargs):
        super().__init__(**kwargs)
        self.target_label = target_label  # Target label to perturb toward

    def self_training_label(self, labels, idx_train):
        # Initialize labels_self_training with the target label for all nodes
        labels_self_training = torch.full_like(labels, self.target_label)
        
        # Retain the true labels for the training set
        labels_self_training[idx_train] = labels[idx_train]
        
        return labels_self_training

    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):
        """Override the meta gradient computation to use the target label."""
        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        # Compute loss for the target label
        target_labels = torch.full_like(labels[idx_unlabeled], fill_value=self.target_label)
        attack_loss = F.nll_loss(output[idx_unlabeled], target_labels)

        print(f'Attack loss (target={self.target_label}): {attack_loss.item()}')

        # Compute gradients
        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

        return adj_grad, feature_grad

class TargetedPGDAttack(PGDAttack):
    #Targeted PGD attack to perturb nodes toward a specific label.

    def __init__(self, target_label, **kwargs):
        super().__init__(**kwargs)
        self.target_label = target_label  # Target label to perturb toward

    def _loss(self, output, labels):
        #Override loss function to use the target label
        target_labels = torch.full_like(labels, self.target_label)
        
        if self.loss_type == "CE":
            loss = -F.nll_loss(output, target_labels)
        
        elif self.loss_type == "CW":
            onehot = utils.tensor2onehot(target_labels)
            
            other_logits = output - 1000 * onehot
            best_other_class = other_logits.argmax(1)
            
            margin = (
                output[torch.arange(len(output)), target_labels] -
                output[torch.arange(len(output)), best_other_class]
            )
            loss = torch.clamp(margin, min=0).mean()  # Negative to maximize margin
        
        return loss