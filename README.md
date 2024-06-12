# Graph-aware Logistic Regression

This repository contains the source code to reproduce experiments in the paper: *Graph as a Feature: A Graph-Aware Non-Neural Model for Node Classification*.

## Usage

The `src/main.py` file can be used with the following parameters:
For a specific model, the `src/main.py` can be used with the following parameters:
```bash
--dataset         Graph dataset {cora, pubmed, ...}
--undirected      If true, force graph to be undirected 
--randomstate     Random state seed (default=8)
--k               Value for k-fold splits
--stratified      If true, perform stratified split in training folds (default=true)
--model           Model name {Diffusion, GCN, ...}
--use_features    [optional] If true, use features as input (for baseline models)
--use_concat      [optional] If true, use concatenation of adajcency and feature matrices as input (for baseline models)
```

For example, running expriment on `Cora` for the Graph-aware Logistic Regression model is done with:
```bash
python main.py --dataset=cora --undirected=true --penalized=true --randomstate=8 --k=8 --stratified=true --model=Logistic_regression --use_concat=true
```

## Datasets 

We use the following graph datasets in our experiments. All datasets are available in the `\data` directory.

|Dataset| $$\|V\|$$ | $\|E\|$ | $L$ | $C$ | $\delta_A$ |
|-------|-----------|---------|-----|-----|------------|
| Cora | 2708 | 10556 | 1433 | 7 | 2.88e-03 |
| Pubmed* | 19717 | 88651 | 500 | 3 | 4.56e-04 |
| Citeseer | 3327 | 9104 | 3703 | 6 | 1.65e-03 |
| Actor | 7600 | 30019 | 932 | 5 | 1.04e-03 |
| CS | 18333 | 163788 | 6805 | 15 | 9.75e-04 |
| Photo | 7650 | 238162 | 745 | 8 | 8.14e-03 |
| Cornell | 183 | 298 | 1703 | 5 | 1.79e-02 |
| Wisconsin | 251 | 515 | 1703 | 5 | 1.64e-02 |
| Wikivitals | 10011 | 824999 | 37845 | 11 | 8.23e-03 |
| Wikivitals-fr | 9945 | 558427 | 28198 | 11 | 5.65e-03 |
| Wikischools | 4403 | 112834 | 20527 | 16 | 5.82e-03 |
| Wikivitals+ | 45149 | 3946850| 85512 | 11 | 1.93e-03 |
| ogbn-arxiv | 169343 | 1166246| 85512 | 40 | 8.14e-05 |