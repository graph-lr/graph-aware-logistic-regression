from abc import ABC, abstractmethod
import numpy as np
import os
import pickle
from scipy import sparse
import time

from sklearn.model_selection import StratifiedKFold, KFold

from sknetwork.data import Bunch

import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor


class BaseDataset:
    """Base class for Dataset."""

    def __init__(self, dataset: str, undirected: bool, random_state: int, k: int, stratified: bool):
        self.random_state = random_state
        self.data = self.get_data(dataset, undirected)
        self.kfolds = self.k_fold(self.data, k, random_state, stratified)
        self.netset = None

    def k_fold(self, data: Data, k: int, random_state: int, stratified: bool = True) -> tuple:
        """Split all data in Data into k folds. Each fold contains train/val/test splits, where val and test sizes equal 1/k.
        
        Parameters
        ----------
        data: Data
            torch.Data wrapper containing graph and feature information.
        k: int
            k in k-folds method.
        random_state: int
            Controls the reproducility.
        stratified: bool
            If True, use stratified kfold.
            
        Returns
        -------
            Tuple of train/val/test indices for each fold.
        """
        if stratified:
            skf = StratifiedKFold(k, shuffle=True, random_state=random_state)
        else:
            skf = KFold(k, shuffle=True, random_state=random_state)

        test_indices, train_indices = [], []
        for _, idx in skf.split(torch.zeros(len(data.x)), data.y):
            test_indices.append(torch.from_numpy(idx).to(torch.long))
        val_indices = [test_indices[i - 1] for i in range(k)]

        for i in range(k):
            train_mask = torch.ones(len(data.x), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

        return (train_indices, test_indices, val_indices)
    
    def get_netset(self, dataset: str, pathname: str, use_cache: bool = True):
        """Get data in Netset format (scipy.sparse CSR matrices). Save data in Bunch format if use_cache is set to False.
        
        Parameters
        ----------
        dataset: str
            Dataset name
        pathname: str
            Path to data.
        use_cache: bool (default=True)
            If True, use cached data (if existing).

        Returns
        -------
            Bunch object.
        """
        if os.path.exists(os.path.join(pathname, dataset)) and use_cache:
            with open(os.path.join(pathname, dataset), 'br') as f:
                graph = pickle.load(f)
            print(f'Loaded dataset from {os.path.join(pathname, dataset)}')
        else:
            print(f'Building netset data...')
            # Convert dataset to NetSet format (scipy CSR matrices)
            graph = self.to_netset(dataset)

            # Save Netset dataset
            with open(os.path.join(pathname, dataset), 'bw') as f:
                pickle.dump(graph, f)
            print(f'Netset data saved in {os.path.join(pathname, dataset)}')
        
        self.netset = graph
        
        return self.netset
    
    def to_netset(self, dataset: str):
        """Convert data into Netset format and return Bunch object."""
       
        if dataset.startswith('ogbn') and isinstance(self.data.edge_index, SparseTensor):
            n = self.data.edge_index.sizes()[0]
            rows, cols, _ = self.data.edge_index.coo()
            data = np.ones(len(rows))
            adjacency = sparse.coo_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(n, n)).tocsr()
            biadjacency = sparse.csr_matrix(np.array(self.data.x))
        else:
            # nodes and edges
            rows = np.asarray(self.data.edge_index[0])
            cols = np.asarray(self.data.edge_index[1])
            data = np.ones(len(rows))
            n = len(self.data.y) #len(set(rows).union(set(cols)))
            adjacency = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            if dataset.startswith('ogbn'):
                biadjacency = sparse.csr_matrix(np.array(self.data.x))
            else:
                biadjacency = sparse.csr_matrix(np.array(self.data.x), dtype=bool)

        # Node information
        labels = np.array(self.data.y)

        graph = Bunch()
        graph.adjacency = adjacency.astype(bool)
        graph.biadjacency = biadjacency
        graph.labels_true = labels

        return graph
    
    def _to_custom_data(self, dataset):
        """Convert Dataset format from Pytorch to a modifiable Data object."""
        data = Data(x=dataset.x,
               edge_index=dataset.edge_index,
               num_classes=dataset.num_classes,
               y=dataset.y,
               train_mask=dataset.train_mask,
               val_mask=dataset.val_mask,
               test_mask=dataset.test_mask)
        
        return data
    

class BaseModel(ABC):
    """Base class for models."""

    def __init__(self, name: str):
        self.name = name
        self.train_loader = None
        self.timeout = time.time() + 60*60*5 # 5-hour limit

    @abstractmethod
    def fit_predict(self, dataset, train_idx: np.ndarray = None):
        pass

    @abstractmethod
    def accuracy(dataset, labels_pred, split, penalized, *args):
        pass
