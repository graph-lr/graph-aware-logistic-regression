import numpy as np
from scipy import sparse

from base import BaseModel
from metric import compute_accuracy

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sknetwork.classification import DiffusionClassifier, NNClassifier
from sknetwork.embedding import Spectral


class Baseline(BaseModel):
    """Baseline model class."""
    def __init__(self, name: str, **kwargs):
        super(Baseline, self).__init__(name)
        if name == 'diffusion':
            self.alg = DiffusionClassifier()
        elif name == 'knn':
            if kwargs.get('embedding_method') == 'true':
                self.alg = NNClassifier(n_neighbors=5, embedding_method=Spectral(30))
            else:
                self.alg = NNClassifier()
        elif name == 'logistic_regression':
            self.alg = LogisticRegression()

    def get_seeds(self, labels_true: np.ndarray, train_idx: np.ndarray) -> dict:
        """Get training seeds in the form of a dictionary.
        
        Parameters
        ----------
        labels_true: np.ndarray
            True node labels.
        train_idx: np.ndarray
            Training indexes.
            
        Returns
        -------
            Dictionary of training seeds. """
        # Training data: corresponds to seed nodes
        training_seeds = {i.item(): labels_true[i] for i in train_idx}
        return training_seeds
    
    def transform_data(self, dataset, **kwargs):
        """Apply transformation on data according to parameters.
        
        Parameters
        ----------
        dataset
            Dataset object.
        
        Returns
        -------
        Transformed data as a sparse matrix.
        """
        # Use only features matrix
        if kwargs.get('use_features') == 'true':
            X = dataset.netset.biadjacency

        # Use concatenation of adjacency and features matrix 
        elif kwargs.get('use_concat') == 'true':
            X = sparse.hstack((dataset.netset.adjacency, dataset.netset.biadjacency))

        # Use only graph structure, i.e. adjacency matrix
        else:
            X = dataset.netset.adjacency

        return X
    
    def fit_predict(self, dataset, train_idx: np.ndarray, val_idx: np.ndarray = None, test_idx : np.ndarray = None, **kwargs) -> np.ndarray:
        """Fit algorithm on training data and predict node labels.
        
        Parameters
        ----------
        dataset
            Dataset object.
        train_idx: np.ndarray
            Training indexes.
            
        Returns
        -------
            Array of predicted node labels.
        """
        # Transform data
        X = self.transform_data(dataset, **kwargs)
        
        # Logistic regression from Sklearn does not have a fit_predict method
        if self.name == 'logistic_regression':
            labels_pred = self.alg.fit(X[train_idx, :],
                                       dataset.netset.labels_true[train_idx]).predict(X)
            
        else:
            training_seeds = self.get_seeds(dataset.netset.labels_true, train_idx) 
            labels_pred = self.alg.fit_predict(X, training_seeds)
        
        # If graph is not connected, predicted label can be -1 (e.g. Diffusion model)
        # In this case, we assign randomly a class to the node
        n_negs = (labels_pred == -1).sum()
        if n_negs > 0:
            n_labels = len(np.unique(dataset.netset.labels_true))
            labels_pred[labels_pred == -1] = np.random.randint(n_labels, size=(labels_pred == -1).sum())

        return labels_pred
    
    def accuracy(self, dataset, labels_pred: np.ndarray, split: np.ndarray, penalized: bool, *args) -> float:
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
            
        Returns
        -------
            Accuracy score"""
        return compute_accuracy(dataset.netset.labels_true[split], labels_pred[split], penalized)
