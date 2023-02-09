# Master-Thesis
Repository containing the code I wrote for my master thesis

## Goal
My master thesis is about making a Distributed Byzantine resilient ADAM optimizer. A central server gathers the first and second moments computed by each participant and aggregates them.
The assumptions are that I do not have any control over the workers participating in the training, nor do I know how many Byzantine workers there are. 
Hence, some workers are either consciously or unconsciously Byzantine. A worker might want to implement backdoors in our model or prevent convergence. However, some workers might have outdated models and send outdated moments. 

I have to study which aggregation function is the most robust and build some mathematical proofs to ensure they converge. 

## Installation

The code works in an conda environment with python 3.10. The requirements can be installed by running `conda create -n <environment-name> --file req.txt`

## Running
1. Call `custom_data_splitting(number_participants=1)` from `data_preparation.py` with the wanted number of participants (I use 15) to create the training sets. 
2. Build the necessary directories that will store the results by calling `build_directories()` in `data_preparation.py`
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
