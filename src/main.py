import argparse
from collections import defaultdict
import numpy as np
import os
from utils import save_dict, check_exists, get_dataset
from train import Trainer


def run(dataset, undirected, penalized, random_state, k, stratified, model, **kwargs):
    """Run experiment."""
    
    outs = defaultdict(list)
    for fold, (train_idx, test_idx, val_idx) in enumerate(zip(*dataset.kfolds)):
        # Model training + predictions
        # In semi-supervised classification, model trains on the entire network (graph + features), but has access only to node labels in the training split. It is then evaluated on (val and) test split, in which labels are unknown.
        trainer = Trainer(train_idx, val_idx, test_idx)
        train_acc, test_acc, elapsed_time = trainer(model, dataset, penalized, **kwargs)

        # OOM triggered: remaining folds are not computed
        if elapsed_time == 'OOM':
           outs['train acc'].append(train_acc)
           outs['test acc'].append(test_acc)
           outs['elapsed_time'].append(elapsed_time)
           return outs
        
        # Save training and test scores for averages on n runs
        outs['train acc'].append(train_acc)
        outs['test acc'].append(test_acc)
        outs['elapsed_time'].append(elapsed_time)

    print('Train acc: ', np.mean(outs['train acc']))
    print('Test acc: ', np.mean(outs['test acc']))
    
    return outs


if __name__=='__main__':
    FORCE_RUN = True # If True, run experiment even if results already exists
    USE_CACHE = False # If True, use cached dataset if existing 
    DATAPATH = os.path.join(os.path.dirname(os.getcwd()), 'data')
    RUNPATH = os.path.join(os.path.dirname(os.getcwd()), 'runs')

    # Parse arguments
    parser = argparse.ArgumentParser()

    # Required
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--undirected', type=str, required=True)
    parser.add_argument('--penalized', type=str, required=True)
    parser.add_argument('--randomstate', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--stratified', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)

    # Optional
    parser.add_argument('--embedding_method', type=str, required=False)
    parser.add_argument('--use_features', type=str, required=False)
    parser.add_argument('--use_concat', type=str, required=False)
    #parser.add_argument('--logger', type=str, default=None)
    args = parser.parse_args()

    # Output filename
    filename = []
    for attr, value in vars(args).items():
        if value is not None:
            if attr not in ['dataset', 'model']:
                filename.append(attr + str(value).lower())
            else:
                filename.append(str(value).lower())
    filename = '_'.join(filename)

    # If run already exists, pass
    check_exists(RUNPATH, filename, force_run=FORCE_RUN)
    
    undirected = args.undirected.lower() == 'true'
    penalized = args.penalized.lower() == 'true'
    stratified = args.stratified.lower() == 'true'

    # Create dataset for GNN based models
    dataset = get_dataset(args.dataset,
                          undirected,
                          args.randomstate,
                          args.k,
                          stratified)
    
    # For baseline models
    netset_dataset = dataset.get_netset(args.dataset, DATAPATH, use_cache=USE_CACHE)

    print(f'Algorithm: {args.model}')
    print(f'Dataset: {args.dataset} (#nodes={dataset.data.x.shape[0]}, undirected: {undirected})')
    print(f'kfolds: {args.k} (stratified: {stratified})')

    # Dictionary of arguments
    kwargs = {
        'dataset': dataset, 
        'undirected': undirected,
        'penalized': penalized,
        'random_state': args.randomstate,
        'k': args.k,
        'stratified': stratified,
        'model': args.model,
        'embedding_method': args.embedding_method,
        'use_features': args.use_features,
        'use_concat': args.use_concat
    }
    
    # Run experiment
    outs = run(**kwargs)

    # Save results
    global_out = {}
    global_out['meta'] = args
    global_out['results'] = outs
    save_dict(RUNPATH, filename, global_out)
    