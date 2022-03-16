from meta_bo.environment import ArgusSimEnvironment
from experiments.util import NumpyArrayEncoder
from meta_bo.domain import DiscreteDomain, ContinuousDomain
from meta_bo.models import GPRegressionVanilla
from meta_bo.algorithms.acquisition import UCB
from config import DATA_DIR

import os
import time
import json
import glob
import numpy as np

STORAGE_DIR_UNIFORM = os.path.join(DATA_DIR, 'uniform_domain_data')
SEED = 6783

def generate_uniform_data(num_samples: int, step_size: float, dump_period=10, seed: int = SEED):
    rds = np.random.RandomState(seed)
    run_id = rds.randint(0, 10**8)
    params_dict = {'stepsize': step_size}
    env = ArgusSimEnvironment(params={'stepsize': step_size}, random_state=rds)

    x = rds.uniform(env.domain.l, env.domain.u, size=(num_samples, env.domain.d))
    evals = []
    t = time.time()
    for i in range(num_samples):
        evals.append(env.evaluate(x[i, :]))
        if (i > 0 and i % dump_period == 0) or (i >= num_samples - 1):
            evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
            evals_dict = {'params': params_dict, 'evals': evals_stacked}

            duration_avg = (time.time() - t) / dump_period
            # store data
            os.makedirs(STORAGE_DIR_UNIFORM, exist_ok=True)
            storage_path = os.path.join(STORAGE_DIR_UNIFORM, f'argus_domain_data_stepsize_{step_size:.6f}_{run_id}.json')
            with open(storage_path, 'w') as f:
                json.dump(evals_dict, f, cls=NumpyArrayEncoder)
            print(f'Sample {i+1}/{num_samples}. Average iter time {duration_avg} sec. Dumped results to %s'%storage_path)
            t = time.time()

def main(args):
    return generate_uniform_data(args.num_samples, args.step_size, dump_period=10, seed=args.seed)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate meta-training data')
    parser.add_argument('--step_size', type=float, default=0.01, help='')
    parser.add_argument('--num_samples', type=int, default=10000, help='')
    parser.add_argument('--seed', type=int, default=SEED, help='random number generator seed')
    args = parser.parse_args()
    main(args)