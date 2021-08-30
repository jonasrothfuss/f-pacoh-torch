from matplotlib import pyplot as plt
import numpy as np
import argparse
import time
import math
import os
import json
from absl import logging

from meta_bo.meta_environment import RandomBraninMetaEnv, RandomMixtureMetaEnv
from meta_bo.algorithms.acquisition import UCB
from meta_bo.models.pacoh_map import PACOH_MAP_GP
from meta_bo.models.f_pacoh_map import FPACOH_MAP_GP
from meta_bo.models.vanilla_gp import GPRegressionVanilla
from experiments.generate_meta_training_data import get_meta_data
from experiments.util import NumpyArrayEncoder

import meta_bo.meta_environment


def main(args):
    t_start = time.time()
    rds = np.random.RandomState(args.seed)

    meta_env_class = getattr(meta_bo.meta_environment, args.env)
    meta_env = meta_env_class(random_state=rds)
    meta_train_data, meta_train_data_q = get_meta_data(args.env, num_tasks=5, num_samples=30, gp=args.gp_data,
                                                       kernel_lengthscale=0.1)

    logging.set_verbosity(logging.INFO)

    if args.model == 'PACOH':
        model = PACOH_MAP_GP(input_dim=meta_env.domain.d, normalization_stats=meta_env.normalization_stats,
                             weight_decay=args.weight_decay,
                             feature_dim=args.feature_dim,
                             num_iter_fit=args.num_iter_fit,
                             lr=args.lr,
                             lr_decay=args.lr_decay,
                             random_state=rds)
    elif args.model == 'FPACOH':
        model = FPACOH_MAP_GP(domain=meta_env.domain, normalization_stats=meta_env.normalization_stats,
                              weight_decay=args.weight_decay,
                              feature_dim=args.feature_dim,
                              num_iter_fit=args.num_iter_fit,
                              lr=args.lr,
                              lr_decay=args.lr_decay,
                              prior_lengthscale=args.prior_lengthscale,
                              prior_outputscale=args.prior_outputscale,
                              num_samples_kl=args.num_samples_kl,
                              prior_factor=args.prior_factor,
                              random_state=rds)
    elif args.model == 'Vanilla_GP':
        model = GPRegressionVanilla(input_dim=meta_env.domain.d, normalization_stats=meta_env.normalization_stats,
                                    kernel_variance=args.kernel_variance,
                                    kernel_lengthscale=args.kernel_lengthscale,
                                    likelihood_std=args.likelihood_std,
                                    random_state=rds)
    else:
        raise NotImplementedError

    if not args.model == 'Vanilla_GP':
        model.meta_fit(meta_train_data, meta_valid_tuples=None, log_period=100)

    t_after_meta_fit = time.time()

    algo = UCB(model, meta_env.domain, beta=2.0)
    evals = []

    env = meta_env.sample_env()

    for t in range(args.steps):
        x = algo.next()

        if args.compute_bp:
            x_bp = algo.best_predicted()
            evaluation = env.evaluate(x, x_bp=x_bp)
        else:
            evaluation = env.evaluate(x)
        evals.append(evaluation)
        algo.add_data(evaluation['x'], evaluation['y'])

    t_after_bo = time.time()

    """ store results """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}

    results_dict = {
        'evals': evals_stacked,
        'params': args.__dict__,
        'duration_model_meta_fit': t_after_meta_fit - t_start,
        'duration_bo': t_after_bo - t_after_meta_fit,
        'duration_total': t_after_meta_fit - t_start
    }

    if args.exp_result_folder is None:
        from pprint import pprint
        pprint(results_dict)
    else:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        exp_hash = str(abs(json.dumps(results_dict['params'], sort_keys=True).__hash__()))
        exp_result_file = os.path.join(args.exp_result_folder, '%s.json'%exp_hash)
        with open(exp_result_file, 'w') as f:
            json.dump(results_dict, f, indent=4, cls=NumpyArrayEncoder)
        print('Dumped results to %s'%exp_result_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-BO run')
    parser.add_argument('--env', type=str, default='RandomMixtureMetaEnv', help='BO environment')
    parser.add_argument('--model', type=str, default='PACOH', help='Meta-Learner for the GP-Prior')
    parser.add_argument('--steps', type=int, default=100, help='Number of BO steps')
    parser.add_argument('--compute_bp', type=bool, default=True, help='whether to compute best predicted')
    parser.add_argument('--gp_data', type=bool, default=False, help='whether to use gp-ucb to collect the data')
    parser.add_argument('--seed', type=int, default=834, help='random number generator seed')

    parser.add_argument('--exp_result_folder', type=str, default=None)

    # model arguments
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--feature_dim', type=int, default=2)
    parser.add_argument('--num_iter_fit', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_decay', type=float, default=1.0)

    parser.add_argument('--prior_lengthscale', type=float, default=0.2)
    parser.add_argument('--prior_outputscale', type=float, default=2.0)
    parser.add_argument('--num_samples_kl', type=int, default=20)
    parser.add_argument('--prior_factor', type=float, default=0.2)

    parser.add_argument('--kernel_variance', type=float, default=1.5)
    parser.add_argument('--kernel_lengthscale', type=float, default=0.2)
    parser.add_argument('--likelihood_std', type=float, default=0.05)

    args = parser.parse_args()
    main(args)