import meta_bo.meta_environment
from meta_bo.domain import DiscreteDomain, ContinuousDomain
from meta_bo.models import GPRegressionVanilla
from meta_bo.algorithms.acquisition import UCB
from config import DATA_DIR

import os
import time
import json
import numpy as np

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def generate_uniform_points(env, num_samples, rds):
    if isinstance(env.domain, ContinuousDomain):
        x = rds.uniform(env.domain.l, env.domain.u,
                                  size=(num_samples, env.domain.d))
    elif isinstance(env.domain, DiscreteDomain):
        x = rds.choice(env.domain.points, num_samples, replace=True)
    else:
        raise NotImplementedError

    evals = []
    for i in range(num_samples):
        evals.append(env.evaluate(x[i, :]))
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    return evals_stacked


def generate_gp_ucb_points(env, num_samples, kernel_lengthscale=0.2, rds=None):
    model = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats,
                                kernel_lengthscale=kernel_lengthscale, normalize_data=True, random_state=rds)
    algo = UCB(model, env.domain, beta=2.0, random_state=rds)
    evals = []
    for t in range(num_samples):
        t = time.time()
        x = algo.next()
        evaluation = env.evaluate(x)
        evals.append(evaluation)
        algo.add_data(evaluation['x'], evaluation['y'])
        step_time = time.time() - t
        print('step %i/%i | x: %s | y %.6f | evaluation time: %.2f sec' % (evaluation['t'], num_samples,
                                                                               str(evaluation['x']),
                                                                               evaluation['y'],
                                                                               step_time))
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    return evals_stacked

def main(args):
    rds = np.random.RandomState(args.seed)
    meta_env_class = getattr(meta_bo.meta_environment, args.env)
    meta_env = meta_env_class(random_state=rds)

    # generate task data
    tasks = []
    for i in range(args.num_tasks):
        print('\nCollecting data for task %i of %i -----' % (i + 1, args.num_tasks))
        params = meta_env.sample_env_param()
        if args.env == 'ArgusSimMetaEnv':
            env = meta_env.env_class(params=params, random_state=meta_env._rds, matlab_engine=meta_env.matlab_engine)
        else:
            env = meta_env.env_class(params=params, random_state=meta_env._rds)
        if args.gp:
            evals_stacked = generate_gp_ucb_points(env, args.num_samples, kernel_lengthscale=0.2, rds=rds)
        else:
            evals_stacked = generate_uniform_points(env, args.num_samples, rds)
        tasks.append({'params': params, 'evals': evals_stacked})

    # store data
    if args.gp:
        storage_dir = os.path.join(DATA_DIR, 'gp_ucb_meta_data')
    else:
        storage_dir = os.path.join(DATA_DIR, 'uniform_meta_data')

    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, '%s_%i_tasks_%i_samples.json'%(args.env, args.num_tasks, args.num_samples))
    with open(storage_path, 'w') as f:
        json.dump(tasks, f, cls=NumpyArrayEncoder)
    print('Dumped results to %s'%storage_path)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate meta-training data')
    parser.add_argument('--gp', action='store_true', help='whether to use gp-ucb to collect the data')
    parser.add_argument('--num_tasks', type=int, default=20, help='number of tasks')
    parser.add_argument('--num_samples', type=int, default=20, help='number of samples per task')
    parser.add_argument('--seed', type=int, default=6783, help='random number generator seed')
    parser.add_argument('--env', type=str, default='ArgusSimMetaEnv', help='Environment from which to collect data')
    args = parser.parse_args()
    main(args)