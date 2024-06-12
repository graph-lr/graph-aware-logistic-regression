from base import BaseModel
import numpy as np
from scipy import sparse
from sknetwork.embedding import Spectral
import time
from tqdm import tqdm

import dgl
from dgl.nn import APPNPConv

import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SGConv
from torch_geometric.nn.models import GCN as GCNTorchModel
import torch.nn.functional as F
from torch_sparse.tensor import SparseTensor

from dataset import WikivitalsDataset
from metric import compute_accuracy

"""
Models developed in PyTorchGeometric library are available here.
For state-of-the-art algorithms not avaiblable in PyTorchGeometric, see sota_models.py file.

Model configurations and parameters are taken from original papers.
"""


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 16):
        super().__init__()
        #torch.manual_seed(0)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        if isinstance(edge_index, SparseTensor):
            # for OGB datasets
            return x.log_softmax(dim=-1)
        else:
            return F.log_softmax(x, dim=1)
    

class GraphSage(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 256):
        super().__init__()
        self.conv1 = SAGEConv(in_channels=dataset.num_features, out_channels=hidden_channels, aggr='max')
        self.conv2 = SAGEConv(hidden_channels, dataset.num_classes, aggr='max')

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        if isinstance(edge_index, SparseTensor):
            # for OGB datasets
            return x.log_softmax(dim=-1)
        else:
            return F.log_softmax(x, dim=1)
    

class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 8):
        super().__init__()
        self.conv1 = GATConv(in_channels=dataset.num_features, out_channels=hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels * 8, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class SGC(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = SGConv(dataset.num_features, dataset.num_classes, K=2, cached=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return F.log_softmax(x, dim=1)
    

class APPNP(torch.nn.Module):
    """APPNP developed using DGL."""
    def __init__(self, dim_input: int, hidden_channels: list, dim_output: int, feat_drop: float = 0.5, edge_drop: float = 0.5, alpha: float = 0.1, k: int = 10):
        super(APPNP, self).__init__()
        self.layers = torch.nn.ModuleList()
        # input layer
        self.layers.append(torch.nn.Linear(dim_input, hidden_channels[0]))
        
        # hidden layers
        for i in range(1, len(hidden_channels)):
            self.layers.append(torch.nn.Linear(hidden_channels[i - 1], hidden_channels[i]))
        
        # output layer
        self.layers.append(torch.nn.Linear(hidden_channels[-1], dim_output))
        self.activation = F.relu
        
        if feat_drop:
            self.feat_drop = torch.nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
            
        self.propagate = APPNPConv(k, alpha, edge_drop)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, graph, features):
        # prediction step
        h = features
        h = self.feat_drop(h)
        h = self.activation(self.layers[0](h))
        for layer in self.layers[1:-1]:
            h = self.activation(layer(h))
        h = self.layers[-1](self.feat_drop(h))
        # propagation step
        h = self.propagate(graph, h)
        return h


class GNN(BaseModel):
    """GNN model class."""
    def __init__(self, name: str, dataset, train_idx: np.ndarray, **kwargs):
        super(GNN, self).__init__(name)

        # Transform features if needed
        dataset = self.transform_data(dataset, **kwargs)

        # Initialize model
        if name == "gcn":
            self.alg = GCN(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = F.nll_loss
            self.n_epochs = 200
        elif name == 'graphsage':
            self.alg = GraphSage(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = F.nll_loss
            self.n_epochs = 10
            self.train_loader = NeighborLoader(dataset.data, num_neighbors=[25, 10],
                                               batch_size=512, input_nodes=train_idx)
        elif name == 'gat':
            self.alg = GAT(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.05, weight_decay=5e-4)
            self.criterion = F.nll_loss
            self.n_epochs = 100
        elif name == 'sgc':
            self.alg = SGC(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.2, weight_decay=0.005)
            self.criterion = F.nll_loss
            self.n_epochs = 100
        elif name == 'jumpingknowledge':
            # https://proceedings.mlr.press/v80/xu18c/xu18c.pdf
            self.alg = GCNTorchModel(in_channels=dataset.data.x.shape[1],
                                     hidden_channels=32,
                                     num_layers=2,
                                     out_channels=dataset.data.num_classes,
                                     dropout=0.5,
                                     jk='cat')
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss() #combines log_softmax and nll_loss as the output of GCNTorchModel is the raw output
            self.n_epochs = 100

    def update_masks(self, data, train_idx, val_idx, test_idx):
        """Update train/val/test mask in Torch Dataset object.
        
        Parameters
        ----------
        data: Torch Dataset
            Torch Dataset object.
        train_idx: np.ndarray
            Training indexes.
        val_idx: np.ndarray
            Validation indexes.
        test_idx: np.ndarray
            Test indexes.
            
        Returns
        -------
            Updated Torch Dataset object.
        """
        # Torch geometric format
        train_mask = torch.zeros(len(data.x), dtype=bool)
        train_mask[train_idx] = True
        data.train_mask = train_mask
        test_mask = torch.zeros(len(data.x), dtype=bool)
        test_mask[test_idx] = True

        if val_idx is not None:
            test_mask[val_idx] = True # ---> no val set
        data.test_mask = test_mask

        return data
    
    def train(self, dataset) -> torch.Tensor:
        """Training function.
        
        Parameters
        ----------
        dataset: Custom Dataset object.
        
        Returns
        -------
            Loss. 
        """
        self.alg.train()
        if self.train_loader is not None:
            total_examples = total_loss = 0
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                batch_size = batch.batch_size
                out = self.alg(batch.x, batch.edge_index)[:batch_size]
                #out = self.alg(batch.x, batch.edge_index)
                #loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
                loss = self.criterion(out, batch.y[:batch_size])
                loss.backward()
                self.optimizer.step()

                #total_loss += loss
                total_examples += batch_size
                total_loss += float(loss) * batch_size
                
            #loss = total_loss / len(self.train_loader)
            loss = total_loss / total_examples
        else:
            self.optimizer.zero_grad()
            out = self.alg(dataset.x, dataset.edge_index)
            loss = self.criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            self.optimizer.step()

        return loss

    def test(self, data):
        """Test function.
        
        Parameters
        ----------
        data: Torch Data object.
        
        Returns
        -------
            Loss.
        """
        self.alg.eval()
        out = self.alg(data.x, data.edge_index)
    
        pred = out.argmax(dim=1)
        test_correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
        
        return test_acc
    
    def transform_data(self, dataset, **kwargs):
        """Apply transformation on data according to parameters.
        
        Parameters
        ----------
        dataset
            Dataset object.
        
        Returns
        -------
        Transformed data.
        """
        # Use concatenation of adjacency and features matrix 
        if kwargs.get('use_concat') == 'true':
            X = torch.tensor(sparse.hstack((dataset.netset.adjacency, dataset.netset.biadjacency)).todense(), dtype=torch.float32)
            dataset.data.x = X
            dataset.num_features = dataset.data.x.shape[1]

        return dataset
    
    def fit_predict(self, dataset, train_idx: np.ndarray, val_idx: np.ndarray, test_idx : np.ndarray, **kwargs) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Custom Dataset object.
            
        Returns
        -------
            Array of predicted node labels.
        """
        # Build train/val/test masks
        dataset.data = self.update_masks(dataset.data, train_idx, val_idx, test_idx)

        # Train model
        for epoch in tqdm(range(1, self.n_epochs + 1)):
            loss = self.train(dataset.data)
            
            # Time constraint
            if time.time() > self.timeout:
                return -1
            
        self.alg.eval()
        out = self.alg(dataset.data.x, dataset.data.edge_index)
        y_pred = out.argmax(dim=1)

        return y_pred
    
    def accuracy(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool, split_name: str) -> float:
        """Accuracy score.
        
        Parameters
        ----------
        dataset
            Dataset object.
        labels_pred: np.ndarray
            Predicted labels.
        split: np.ndarray
            Split indexes.
        penalized: bool
            If true, labels not predicted (with value -1) are considered in the accuracy computation.
        split_name: str
            Either 'train' or 'test'.
            
        Returns
        -------
            Accuracy score"""
        if split_name == 'train':
            return compute_accuracy(np.array(dataset.data.y[dataset.data.train_mask]), np.array(labels_pred[dataset.data.train_mask]), penalized)
        elif split_name == 'test':
            return self.test(dataset.data)
        
    def accuracy_degree(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool, split_name: str) -> float:
        """Accuracy score on nodes grouped by node degree.
        
        Parameters
        ----------
        dataset
            Dataset object.
        labels_pred: np.ndarray
            Predicted labels.
        split: np.ndarray
            Split indexes.
        penalized: bool
            If true, labels not predicted (with value -1) are considered in the accuracy computation.
        split_name: str
            Either 'train' or 'test'.
            
        Returns
        -------
            Dictionary of accuracy scores for each node group considering their degrees."""
        if split_name != 'test':
            raise Exception('Should only be computed on test nodes.')
        else:
            mask = dataset.data.test_mask
            adj = dataset.netset.adjacency
            acc_groups = {}

            
            in_degrees = torch.Tensor(adj.T.dot(np.ones(adj.shape[1])))
            for deg in np.unique(in_degrees):
                mask_with_degs = (in_degrees == deg) & mask
                
                # Minimum number of nodes per category is set to 1
                if mask_with_degs.sum() > 1:
                    self.alg.eval()
                    out = self.alg(dataset.data.x, dataset.data.edge_index)
                    labels_pred = out.argmax(dim=1)
                    test_correct = labels_pred[mask_with_degs] == dataset.data.y[mask_with_degs]
                    acc_group = int(test_correct.sum()) / int(mask_with_degs.sum())
                    acc_groups[deg] = acc_group
        
        return acc_groups


class GNNDGLBase(BaseModel):
    """Base class for GNN models developed with DGL."""
    def __init__(self, name: str, dataset, train_idx: np.ndarray):
        super(GNNDGLBase, self).__init__(name)

        if name == 'appnp':
            # https://arxiv.org/pdf/1810.05997.pdf
            # https://github.com/dmlc/dgl/blob/master/examples/pytorch/appnp/appnp.py
            # Transform data to DGL format
            self.adjacency = dgl.from_scipy(dataset.netset.adjacency)
            # Embedding for wiki features
            if isinstance(dataset, WikivitalsDataset):
                spectral = Spectral(100)
                features = spectral.fit_transform(dataset.netset.biadjacency)
                self.features = torch.Tensor(features)
            else:
                self.features = dataset.data.x
            self.labels = torch.Tensor(dataset.netset.labels_true).long()
            dim_hidden = [64]
            self.alg = APPNP(self.features.shape[1], dim_hidden, dataset.data.num_classes)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = torch.nn.CrossEntropyLoss()
            self.n_epochs = 200
        else:
            print('Unknow model')

    def update_masks(self, data, train_idx, val_idx, test_idx):
        """From train, val, test indexes, get train, val, test tensor masks.
        
        Parameters
        ----------
        data: Torch Dataset
            Torch Dataset object.
        train_idx: np.ndarray
            Training indexes.
        val_idx: np.ndarray
            Validation indexes.
        test_idx: np.ndarray
            Test indexes.
            
        Returns
        -------
            Tensor masks.
        """
        train_mask = torch.zeros(len(data.x), dtype=bool)
        train_mask[train_idx] = True
        
        test_mask = torch.zeros(len(data.x), dtype=bool)
        test_mask[test_idx] = True

        if val_idx is not None:
            test_mask[val_idx] = True # ---> no val set

        return train_mask, test_mask
    
    def test(self, graph, features, labels, mask):
        """Test function.
        
        Parameters
        ----------
        data: Torch Data object.
        
        Returns
        -------
            Loss.
        """
        self.alg.eval()
        with torch.no_grad():
            output = self.alg(graph, features)
            labels_pred = torch.max(output, dim=1)[1]
            #score = np.mean(np.array(labels[mask]) == np.array(labels_pred[mask]))

        return labels_pred
    
    def fit_predict(self, dataset, train_idx: np.ndarray, val_idx: np.ndarray, test_idx : np.ndarray, **kwargs) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Custom Dataset object.
            
        Returns
        -------
            Array of predicted node labels.
        """
        # Build train/val/test masks
        self.train_mask, self.test_mask = self.update_masks(dataset.data, train_idx, val_idx, test_idx)

        # Train model
        self.alg.train()

        for epoch in range(1, self.n_epochs + 1):
            # forward
            output = self.alg(self.adjacency, self.features)
            loss = self.criterion(output[self.train_mask], self.labels[self.train_mask])

            # backward
            self.optimizer.zero_grad()
            loss.backward()            
            self.optimizer.step()

            # Time constraint
            if time.time() > self.timeout:
                return -1

        labels_pred = self.test(self.adjacency, self.features, self.labels, self.train_mask)
        
        return labels_pred
    
    def accuracy(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool, split_name: str) -> float:
        """Accuracy score.
        
        Parameters
        ----------
        dataset
            Dataset object.
        labels_pred: np.ndarray
            Predicted labels.
        split: np.ndarray
            Split indexes.
        penalized: bool
            If true, labels not predicted (with value -1) are considered in the accuracy computation.
        split_name: str
            Either 'train' or 'test'.
            
        Returns
        -------
            Accuracy score"""
        if split_name == 'train':
            mask = self.train_mask
        else:
            mask = self.test_mask

        labels_pred = self.test(self.adjacency, self.features, self.labels, mask)
        acc = compute_accuracy(np.array(self.labels[mask]), np.array(labels_pred[mask]), penalized)
        
        return acc
