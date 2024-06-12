import numpy as np
import os
import pickle
import sys

from base import BaseModel, BaseDataset
from baseline import Baseline
from dataset import PlanetoidDataset, WikivitalsDataset, CoAuthorDataset, \
    ActorDataset, AmazonDataset, WebKBDataset, PubMedDataset, OGBDataset
from model import GNN, GNNDGLBase
from sota_model import SOTAGNN


def get_dataset(dataset: str, undirected: bool, random_state: int, k: int, stratified: bool) -> BaseDataset:
    """Get Dataset."""
    if dataset.lower() in ['cora', 'citeseer']:
        return PlanetoidDataset(dataset, undirected, random_state, k, stratified)
    # Pubmed from torch_geometric seems to have its features shuffled. We build the graph from sources and import it form netset.
    elif dataset.lower() in ['pubmed']:
        return PubMedDataset(dataset, undirected, random_state, k, stratified)
    elif dataset.lower() in ['wikivitals', 'wikivitals-fr', 'wikischools', 'wikivitals+']:
        return WikivitalsDataset(dataset, undirected, random_state, k, stratified)
    elif dataset.lower() in ['cs']:
        return CoAuthorDataset(dataset, undirected, random_state, k, stratified)
    elif dataset.lower() in ['actor']:
        return ActorDataset(dataset, undirected, random_state, k, stratified)
    elif dataset.lower() in ['photo']:
        return AmazonDataset(dataset, undirected, random_state, k, stratified)
    elif dataset.lower() in ['cornell', 'wisconsin']:
        return WebKBDataset(dataset, undirected, random_state, k, stratified)
    elif dataset.lower() in ['ogbn-mag', 'ogbn-products', 'ogbn-arxiv']:
        return OGBDataset(dataset, undirected, random_state, k, stratified)
    else:
        raise ValueError(f'Unknown dataset: {dataset}.')

def get_model(model: str, dataset = None, train_idx : np.ndarray = None, **kwargs) -> BaseModel:
    """Get model."""
    if model.lower() in ['diffusion', 'knn', 'logistic_regression']:
        return Baseline(model.lower(), **kwargs)
    # models developped in torch_geometric
    elif model.lower() in ['gcn', 'graphsage', 'gat', 'sgc', 'jumpingknowledge']:
        return GNN(model.lower(), dataset, train_idx, **kwargs)
    # Source code from GitHub
    elif model.lower() in ['gcnii', 'h2gcn']:
        return SOTAGNN(model.lower(), dataset, train_idx)
    # SOTA models developped in DGL
    elif model.lower() in ['appnp']:
        return GNNDGLBase(model.lower(), dataset, train_idx)
    else:
        raise ValueError(f'Unknown model: {model}.')
    
def save_dict(path: str, filename: str, data: dict):
    """Save dictionary"""
    with open(f'{os.path.join(path, filename)}', 'wb') as f:
        pickle.dump(data, f)

def load_dict(path: str, filename: str) -> dict:
    """Load dictionary."""
    with open(f'{os.path.join(path, filename)}', 'rb') as f:
        data = pickle.load(f)
    return data

def check_exists(path: str, filename: str, force_run: bool = False):
    """Terminate program if file exists."""
    if not force_run and os.path.exists(os.path.join(path, filename)):
        sys.exit(f'File "{filename}" already exists.')
