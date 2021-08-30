import meta_bo.meta_environment
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

STORAGE_DIR_GP = os.path.join(DATA_DIR, 'gp_ucb_meta_data')
STORAGE_DIR_UNIFORM = os.path.join(DATA_DIR, 'uniform_meta_data')
SEED = 6783

def generate_uniform_points(env, num_samples, rds):
    print(env._params)
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

def generate_gp_ucb_points(env, num_samples, kernel_lengthscale=0.1, rds=None):
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


""" load meta data """

def load_meta_data(path):
    with open(path, 'r') as f:
        meta_data_json = json.load(f)
    has_q = all(['q' in task_dict['evals'] for task_dict in meta_data_json])
    meta_data_target, meta_data_q = [], []
    for task_dict in meta_data_json:
        task_evals = task_dict['evals']
        x = np.array(task_evals['x'])
        y = np.array(task_evals['y'])
        meta_data_target.append((x, y))
        if has_q:
            q = np.array(task_evals['q'])
            meta_data_q.append((x, q))

    if has_q:
        return meta_data_target, meta_data_q
    else:
        return meta_data_target, None

def get_meta_data(env, num_tasks, num_samples, gp=True, kernel_lengthscale=0.2):
    storage_dir = STORAGE_DIR_GP if gp else STORAGE_DIR_UNIFORM
    files = glob.glob(os.path.join(storage_dir, '%s_*_tasks_*samples.json'%env))
    filtered_files = [file for file in files if int(file.split('_')[-4]) >= num_tasks and int(file.split('_')[-2]) >= num_samples]
    if len(filtered_files) > 0:
        return load_meta_data(filtered_files[0])
    else:
        meta_data_file = generate_meta_data(env, num_tasks, num_samples, gp=gp)
        return load_meta_data(meta_data_file)


def generate_meta_data(env_name, num_tasks, num_samples, gp=True, kernel_lengthscale=0.2, seed=SEED):
    rds = np.random.RandomState(seed)
    meta_env_class = getattr(meta_bo.meta_environment, env_name)
    meta_env = meta_env_class(random_state=rds)

    # generate task data
    tasks = []
    for i in range(num_tasks):
        print('\nCollecting data for task %i of %i -----' % (i + 1, num_tasks))
        params = meta_env.sample_env_param()
        if env_name == 'ArgusSimMetaEnv':
            env = meta_env.env_class(params=params, random_state=meta_env._rds, matlab_engine=meta_env.matlab_engine)
        else:
            env = meta_env.env_class(params=params, random_state=meta_env._rds)
        if gp:
            evals_stacked = generate_gp_ucb_points(env, num_samples, kernel_lengthscale=kernel_lengthscale, rds=rds)
        else:
            evals_stacked = generate_uniform_points(env, num_samples, rds)
        tasks.append({'params': params, 'evals': evals_stacked})

    # store data
    storage_dir = STORAGE_DIR_GP if gp else STORAGE_DIR_UNIFORM
    os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, '%s_%i_tasks_%i_samples.json'%(env_name, num_tasks, num_samples))
    with open(storage_path, 'w') as f:
        json.dump(tasks, f, cls=NumpyArrayEncoder)
    print('Dumped results to %s'%storage_path)
    return storage_path

def main(args):
    return generate_meta_data(args.env, args.num_tasks, args.num_samples, gp=args.gp, seed=args.seed)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate meta-training data')
    parser.add_argument('--gp', action='store_true', help='whether to use gp-ucb to collect the data')
    parser.add_argument('--num_tasks', type=int, default=20, help='number of tasks')
    parser.add_argument('--num_samples', type=int, default=20, help='number of samples per task')
    parser.add_argument('--seed', type=int, default=SEED, help='random number generator seed')
    parser.add_argument('--env', type=str, default='RandomMixtureMetaEnv', help='Environment from which to collect data')
    args = parser.parse_args()
    main(args)