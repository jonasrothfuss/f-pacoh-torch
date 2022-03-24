# Meta-BO based on F-PACOH
First and foremost, this repository aims to provide a lightweight and accessible implementation of the F-PACOH algorithm
presented in the paper [*Meta-Learning Reliable Priors in the Function Space*](https://arxiv.org/abs/2106.03195) by J. Rothfuss, D. Heyn, J. Chen and A. Krause.
The comprehensive and much more complex code which was used for the paper's experiments can be downloaded [here](https://www.dropbox.com/sh/n2thesjq87sh66j/AACg-HKMl1NhQpaMOHvvUEOfa?dl=0).

In particular, this repository implements the UCB Bayesian optimization algorithm with support for the following models
* Vanilla GP with SE kernel
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


## Basic Usage
First, add the root of this repository to PYTHONPATH. 
The following code snippet provides a basic usage example of F-PACOH:

```python
import numpy as np
from meta_bo.models.f_pacoh_map import FPACOH_MAP_GP

# generate meta-train data
from meta_bo.meta_environment import RandomMixtureMetaEnv
meta_env = RandomMixtureMetaEnv()
meta_train_tuples = meta_env.generate_uniform_meta_train_data(num_tasks=20, num_points_per_task=10)

# setup and meta-train F-PACOH
fpacoh_model = FPACOH_MAP_GP(domain=meta_env.domain, num_iter_fit=6000, weight_decay=1e-4,
                             prior_factor=0.1, task_batch_size=5)
fpacoh_model.meta_fit(meta_train_tuples)

# make prediction on a new task
test_env = meta_env.sample_env()  # sample a new task from meta_env
x_train, y_train = test_env.generate_uniform_data(num_points=5)  # generate train points
fpacoh_model.add_data(x_train, y_train)  # add train points to model

x_pred = np.linspace(test_env.domain.l, test_env.domain.u, num=100)
posterior_mean, posterior_std = fpacoh_model.predict(x_pred)
```

## Demo notebooks and scripts

We provide also two demo jupyter notebooks for getting started.
* [demo.ipynb](demo.ipynb): A simple, visual demo of Vanilla GPs and F-PACOH for 1d regression.
* [demo.ipynb](demo.ipynb): A visual demo of 1d Bayesian Optimization problem with Vanilla GPs and F-PACOH + the UCB acquisition algorithm.

To run a simple demonstration of GP-UCB with a Vanilla GP model, run the following command:

In addition, we provide simple executable demo scripts for UCB on a 2d function. To run GP-UCB with a vanilla GP, use the following command:
```bash
python experiments/simple_gp_ucb.py
``` 

To run meta-learning of the GP prior with F-PACOH + GP-UCB, run the following command:

```bash
python experiments/meta_gp_ucb.py
``` 

## Citing
For usage of the algorithms provided in this repo for your research,
we kindly ask you to acknowledge the two papers that formally introduce them:
```bibtex
@InProceedings{rothfuss21pacoh,
  title = 	 {PACOH: Bayes-Optimal Meta-Learning with PAC-Guarantees},
  author =       {Rothfuss, Jonas and Fortuin, Vincent and Josifoski, Martin and Krause, Andreas},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9116--9126},
  year = 	 {2021}
 }

@InProceedings{rothfuss2021fpacoh,
  title={Meta-learning reliable priors in the function space},
  author={Rothfuss, Jonas and Heyn, Dominique and Chen, Jinfan and Krause, Andreas},
  booktitle={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```