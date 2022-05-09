# Description
This is the official code repository for the following paper accepted at TNNLS 2022:

Jiang, Y., Wang, S., Valls, V., Ko, B. J., Lee, W. H., Leung, K. K., & Tassiulas, L. (2022). Model pruning enables efficient federated learning on edge devices. IEEE Transactions on Neural Networks and Learning Systems.

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
