from experiments.util import generate_base_command, generate_run_commands, hash_dict
from config import RESULT_DIR

import experiments.run_meta_gp_ucb
import argparse
import numpy as np
import copy
import json
import os
import itertools

applicable_configs = {
    'PACOH': ['weight_decay', 'feature_dim', 'num_iter_fit', 'lr', 'lr_decay'],
    'FPACOH': ['weight_decay', 'feature_dim', 'num_iter_fit', 'lr', 'lr_decay',
               'prior_lengthscale', 'prior_outputscale', 'num_samples_kl', 'prior_factor'],
    'Vanilla_GP': ['kernel_variance', 'kernel_lengthscale', 'likelihood_std']
}

default_configs = {
    # PACOH + FPACOH
    'weight_decay': 1e-4,
    'feature_dim': 2,
    'num_iter_fit': 5000,
    'lr': 1e-3,
    'lr_decay': 1.0,
    # FPACOH
    'prior_lengthscale': 0.2,
    'prior_outputscale': 2.0,
    'num_samples_kl': 20,
    'prior_factor': 0.2,
    # Vanilla_GP
    'kernel_variance': 1.5,
    'kernel_lengthscale': 0.2,
    'likelihood_std': 0.05
}

search_ranges = {
    # PACOH + FPACOH
    'weight_decay': ['loguniform', [-5, -1]],
    'feature_dim': ['choice', [2, 3]],
    'num_iter_fit': ['choice', [4000, 8000]],
    'lr': ['loguniform', [-3.5, -2.5]],
    # FPACOH
    'prior_lengthscale': ['uniform', [0.1, 1.0]],
    'prior_outputscale': ['uniform', [1., 2.]],
    'prior_factor': ['loguniform', [-4, 1.]],
    # Vanilla GP
    'kernel_variance': ['uniform', [1.0, 2.0]],
    'kernel_lengthscale': ['uniform', [0.1, 1.0]],
    'likelihood_std': ['loguniform', [-2.5, -1]],
}

# check consistency of configuration dicts
assert set(itertools.chain(*list(applicable_configs.values()))) == {*default_configs.keys(), *search_ranges.keys()}

def sample_flag(sample_spec, rds=None):
    if rds is None:
        rds = np.random
    assert len(sample_spec) == 2

    sample_type, range = sample_spec
    if sample_type == 'loguniform':
        assert len(range) == 2
        return 10**rds.uniform(*range)
    elif sample_type == 'uniform':
        assert len(range) == 2
        return rds.uniform(*range)
    elif sample_type == 'choice':
        return rds.choice(range)
    else:
        raise NotImplementedError

def main(args):
    rds = np.random.RandomState(args.seed)
    assert args.num_seeds_per_haparam < 100
    init_seeds = list(rds.randint(0, 10**6, size=(100,)))

    # determine name of experiment
    exp_base_path = os.path.join(RESULT_DIR, args.exp_name)
    exp_path = os.path.join(exp_base_path, '%s_%s_%s'%(args.env, args.model, 'gp' if args.gp_data else 'uniform'))


    command_list = []
    for _ in range(args.num_hparam_samples):
        # transfer flags from the args
        flags = copy.deepcopy(args.__dict__)
        [flags.pop(key) for key in ['seed', 'num_hparam_samples', 'num_seeds_per_haparam', 'exp_name', 'num_cpus']]

        # randomly sample flags
        for flag in default_configs:
            if flag in search_ranges:
                flags[flag] = sample_flag(sample_spec=search_ranges[flag], rds=rds)
            else:
                flags[flag] = default_configs[flag]

        # determine subdir which holds the repetitions of the exp
        flags_hash = hash_dict(flags)
        flags['exp_result_folder'] = os.path.join(exp_path, flags_hash)

        for j in range(args.num_seeds_per_haparam):
            seed = init_seeds[j]
            cmd = generate_base_command(experiments.run_meta_gp_ucb, flags=dict(**flags, **{'seed': seed}))
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='local_async', promt=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-BO run')
    parser.add_argument('--env', type=str, default='RandomMixtureMetaEnv', help='BO environment')
    parser.add_argument('--model', type=str, default='Vanilla_GP', help='Meta-Learner for the GP-Prior')
    parser.add_argument('--steps', type=int, default=50, help='Number of BO steps')
    parser.add_argument('--compute_bp', type=bool, default=True, help='whether to compute best predicted')
    parser.add_argument('--gp_data', type=bool, default=False, help='whether to use gp-ucb to collect the data')

    parser.add_argument('--exp_name', type=str, required=True, default=None)

    parser.add_argument('--num_cpus', type=int, default=2, help='random number generator seed')

    parser.add_argument('--seed', type=int, default=382, help='random number generator seed')
    parser.add_argument('--num_hparam_samples', type=int, default=10)
    parser.add_argument('--num_seeds_per_haparam', type=int, default=3)

    args = parser.parse_args()
    main(args)


