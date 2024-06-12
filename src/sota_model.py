from base import BaseModel
from typing import Tuple, Optional, Union
import math
import numpy as np
from scipy import sparse
import time

import torch
from torch import Tensor
import torch_sparse
from torch import FloatTensor
from torch.nn import Parameter, Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul_

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot

from metric import compute_accuracy


# Source code for GCNII model and layer: https://github.com/chennnM/GCNII/blob/master/PyG/cora/cora.py
Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul_(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul_(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNIIdenseConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[torch.Tensor, torch.Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = True,
                 add_self_loops: bool = True, normalize: bool = True,
                 **kwargs):

        super(GCNIIdenseConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, alpha, h0, beta,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        support = (1-beta)*(1-alpha)*x + beta*torch.matmul(x, self.weight1)
        initial = (1-beta)*(alpha)*h0 + beta*torch.matmul(h0, self.weight2)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=support, edge_weight=edge_weight,
                             size=None)+initial
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        assert edge_weight is not None
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
    

class GCNII(torch.nn.Module):
    def __init__(self, dataset, hidden_channels: int = 64):
        super(GCNII, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(dataset.num_features, hidden_channels))

        # from original paper
        self.dropout = 0.5
        self.n_layers = 64
        self.alpha = 0.1
        self.lambda_ = 0.5

        for _ in range(self.n_layers):
            self.convs.append(GCNIIdenseConv(hidden_channels, hidden_channels))
        self.convs.append(torch.nn.Linear(hidden_channels, dataset.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())

    def forward(self, x, edge_index):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        _hidden = []
        x = F.dropout(x, self.dropout ,training=self.training)
        x = F.relu(self.convs[0](x))
        _hidden.append(x)
        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout ,training=self.training)
            beta = math.log(self.lambda_ / (i + 1) + 1)
            #x = F.relu(con(x, edge_index, self.alpha, _hidden[0],beta,edge_weight))
            x = F.relu(con(x, edge_index, self.alpha, _hidden[0], beta))
        x = F.dropout(x, self.dropout ,training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)
    

# Source code for DAGNN: https://github.com/divelab/DeeperGNN/blob/master/DeeperGNN/dagnn.py
def gcn_norm_dagnn(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.
    num_nodes = int(edge_index.max()) + 1 if num_nodes is None else num_nodes
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class Prop(MessagePassing):
    def __init__(self, num_classes: int, K: int, bias: bool=True, **kwargs):
        super(Prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.proj = Linear(num_classes, 1)
        
    def forward(self, x, edge_index, edge_weight=None):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        edge_index, norm = gcn_norm_dagnn(edge_index, edge_weight, x.size(0), dtype=x.dtype)

        preds = []
        preds.append(x)
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            preds.append(x)
           
        pps = torch.stack(preds, dim=1)
        retain_score = self.proj(pps)
        retain_score = retain_score.squeeze()
        retain_score = torch.sigmoid(retain_score)
        retain_score = retain_score.unsqueeze(1)
        out = torch.matmul(retain_score, pps).squeeze()
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)
    
    def reset_parameters(self):
        self.proj.reset_parameters()
    

class H2GCN(torch.nn.Module):
    """H2GCN from https://github.com/GitEventhandler/H2GCN-PyTorch/blob/master/model.py"""
    def __init__(
            self,
            feat_dim: int,
            hidden_dim: int,
            class_dim: int,
            k: int = 2,
            dropout: float = 0.5,
            use_relu: bool = True
    ):
        super(H2GCN, self).__init__()
        self.dropout = dropout
        self.k = k
        self.act = F.relu if use_relu else lambda x: x
        self.use_relu = use_relu
        self.w_embed = torch.nn.Parameter(
            torch.zeros(size=(feat_dim, hidden_dim)),
            requires_grad=True
        )
        self.w_classify = torch.nn.Parameter(
            torch.zeros(size=((2 ** (self.k + 1) - 1) * hidden_dim, class_dim)),
            requires_grad=True
        )
        self.params = [self.w_embed, self.w_classify]
        self.initialized = False
        self.a1 = None
        self.a2 = None
        self.reset_parameter()

    def reset_parameter(self):
        torch.nn.init.xavier_uniform_(self.w_embed)
        torch.nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        return torch.sparse_coo_tensor(
            indices=csp.indices(),
            values=torch.where(csp.values() > 0, 1, 0),
            size=csp.size(),
            dtype=torch.float
        )

    @staticmethod
    def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
        assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
        sp1, sp2 = sp1.coalesce(), sp2.coalesce()
        index1, value1 = sp1.indices(), sp1.values()
        index2, value2 = sp2.indices(), sp2.values()
        m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
        indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
        return torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(m, k),
            dtype=torch.float
        )

    @classmethod
    def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        n = adj.size(0)
        d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
        d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
        d_tiled = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=d_diag,
            size=(n, n)
        )
        return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

    def _prepare_prop(self, adj):
        n = adj.size(0)
        self.initialized = True
        sp_eye = torch.sparse_coo_tensor(
            indices=[list(range(n)), list(range(n))],
            values=[1.0] * n,
            size=(n, n),
            dtype=torch.float
        )
        # initialize A1, A2
        a1 = self._indicator(adj - sp_eye)
        a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)        
        # norm A1 A2
        self.a1 = self._adj_norm(a1)
        self.a2 = self._adj_norm(a2)

    def forward(self, x: FloatTensor, adj: torch.sparse.Tensor) -> FloatTensor:
        # transform adjacency
        adj = self.eidx_to_sp(len(x), adj)

        if not self.initialized:
            self._prepare_prop(adj)
        # H2GCN propagation
        rs = [self.act(torch.mm(x, self.w_embed))]
        for i in range(self.k):
            r_last = rs[-1]
            r1 = torch.spmm(self.a1, r_last)
            r2 = torch.spmm(self.a2, r_last)
            rs.append(self.act(torch.cat([r1, r2], dim=1)))
        r_final = torch.cat(rs, dim=1)
        r_final = F.dropout(r_final, self.dropout, training=self.training)
        #return torch.softmax(torch.mm(r_final, self.w_classify), dim=1)
        return F.log_softmax(torch.mm(r_final, self.w_classify), dim=1)
    
    def eidx_to_sp(self, n: int, edge_index: torch.Tensor) -> torch.sparse.Tensor:
        if isinstance(edge_index, SparseTensor):
            rows, cols, _ = edge_index.coo()
            indices = torch.vstack((rows, cols))
            values = torch.FloatTensor([1.0] * len(rows))
            coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
        else:
            indices = edge_index
            values = torch.FloatTensor([1.0] * len(edge_index[0]))
            coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
        return coo


class SOTAGNN(BaseModel):
    """SOTA GNN model class."""
    def __init__(self, name: str, dataset, train_idx: np.ndarray, **kwargs):
        super(SOTAGNN, self).__init__(name)

        # Transform features if needed
        dataset = self.transform_data(dataset, **kwargs)

        # Initialize model
        if name.lower() == 'gcnii':
            self.alg = GCNII(dataset.data)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            #self.criterion = F.nll_loss()
            self.criterion  = F.nll_loss
            self.n_epochs = 100
        elif name == 'h2gcn':
            # https://github.com/GitEventhandler/H2GCN-PyTorch/blob/master/train.py
            self.alg = H2GCN(feat_dim=dataset.data.x.shape[1],
                             hidden_dim=32,
                             class_dim=dataset.data.num_classes,
                             use_relu=True)
            self.optimizer = torch.optim.Adam(self.alg.parameters(), lr=0.01, weight_decay=5e-4)
            self.criterion = F.nll_loss
            self.n_epochs = 500

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
            total_loss = 0
            for i, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                out = self.alg(batch.x, batch.edge_index)
                loss = self.criterion(out[batch.train_mask], batch.y[batch.train_mask])
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            loss = total_loss / len(self.train_loader)
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
        for epoch in range(1, self.n_epochs + 1):
            loss = self.train(dataset.data)
            # Time constraint
            if time.time() > self.timeout:
                return -1
            
            #train_correct = pred[dataset.train_mask] == dataset.y[dataset.train_mask]
            #train_acc = int(train_correct.sum()) / int(dataset.train_mask.sum())
        #test_acc = self.test(self.alg, dataset)
        out = self.alg(dataset.data.x, dataset.data.edge_index)
        pred = out.argmax(dim=1)

        return pred
    
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
