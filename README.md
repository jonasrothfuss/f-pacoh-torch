# Meta-BO based on PACOH / F-PACOH
The repository implements the UCB Bayesian optimization algorithm with support for the following models
* Vanilla GP
* PACOH-GP [(https://arxiv.org/abs/2002.05551)](https://arxiv.org/abs/2002.05551)
* F-PACOH-GP [(https://arxiv.org/abs/2106.03195)](https://arxiv.org/abs/2106.03195)

## Installation
To install the minimal dependencies needed to use the meta-learning algorithms, run in the main directory of this repository
```bash
pip install .
``` 

For full support of all scripts in the repository, for instance to reproduce the experiments, further dependencies need to be installed. 
To do so, please run in the main directory of this repository 
```bash
pip install -r requirements.txt
``` 


## Usage
First, add the root of this repository to PYTHONPATH.

To run a simple demonstration of GP-UCB with a Vanilla GP model, run the following command:

```bash
python experiments/simple_gp_ucb.py
``` 

To run meta-learning of the GP prior with PACOH / F-PACOH + GP-UCB, run the following command:

```bash
python experiments/meta_gp_ucb.py
``` 