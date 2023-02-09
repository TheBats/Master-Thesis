# Master-Thesis
Repository containing the code I wrote for my master thesis

## Goal
My master thesis is about making a Distributed Byzantine resilient ADAM optimizer. A central server gathers the first and second moments computed by each participant and aggregates them.
The assumptions are that I do not have any control over the workers participating in the training, nor do I know how many Byzantine workers there are. 
Hence, some workers are either consciously or unconsciously Byzantine. A worker might want to implement backdoors in our model or prevent convergence. However, some workers might have outdated models and send outdated moments. 

I have to study which aggregation function is the most robust and build some mathematical proofs to ensure they converge. 

## Installation

The code works in an conda environment with python 3.10. 
The requirements can be installed by running `conda create -n <environment-name> --file spec-file.txt`

Conda is available here: https://docs.conda.io/projects/conda/en/latest/user-guide/install
## Running
1. Create the training sets
`$python ./data_preparation.py create-splits n_participants=15`
2. Build the result directories
`$python ./data_preparation.py build-directories`
3. Run `main.py`. There are currently 10 parameters one can change; learning rate (lr), the number of workers, rounds, attack, number of attackers (f), optimizer, loss, aggregattion method, save.

## Currently implemented aggregation methods and attacks

### Attacks

- Label Flipping
- Sign Flipping
- Little 

### Aggregation

- Average
- Coordinate-Wise Trimmed Mean
- Mean around Median
- Coordinate-Wise Median
