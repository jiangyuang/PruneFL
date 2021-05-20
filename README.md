# Description
This is the official code repository for the following paper:

@article{DBLP:journals/corr/abs-1909-12326,
  author    = {Yuang Jiang and
               Shiqiang Wang and
               Bong{-}Jun Ko and
               Wei{-}Han Lee and
               Leandros Tassiulas},
  title     = {Model Pruning Enables Efficient Federated Learning on Edge Devices},
  journal   = {CoRR},
  volume    = {abs/1909.12326},
  year      = {2019},
  url       = {http://arxiv.org/abs/1909.12326},
  archivePrefix = {arXiv},
  eprint    = {1909.12326},
  timestamp = {Mon, 22 Mar 2021 18:51:05 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1909-12326.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

This repository has some overlap with our model pruning library (https://github.com/jiangyuang/ModelPruningLibrary). However, this repository is for reproducing experiments from the paper only. We will not update this repository along with the model pruning library.

# Setup
```python3
sudo -E python3 setup.py install
```

# Run prototype 
For each new terminal, please run
```shell
source setenv.sh
```
in the `PruneFL` root folder for the correct environment.

To run the prototype, first we need to update the `configs/network.py` configuration with appropriate address and port.

On the server side, run
```python3
# conventional FL
python3 experiments/FEMNIST/prototype/server.py -na -ni
```
for conventional FL, or run
```python3
# PruneFL
python3 experiments/FEMNIST/prototype/server.py -a -i
```
for PruneFL.

On each client side, always run
```python3
python3 experiments/FEMNIST/prototype/client.py
```

# Run simulations
For each new terminal, please run
```shell
source setenv.sh
```
in the `PruneFL` root folder for the correct environment.

To auto-run all experiments, use
```shell
sh autorun/{experiment_name}.sh
```
to run all experiments for `{experiment_name}` (replace by the correct name).

We can also run single experiments using commands in the shell scripts.

# Analyze results
Run
```python3
python3 analysis.py
```
to generate figures in `results/{experiment_name}/figs` folder for each experiment. Non-existing results will be skipped.

The code has been tested on Ubuntu 20.04, and example results are given in the `example_results` folder.
