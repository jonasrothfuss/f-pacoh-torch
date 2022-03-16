from experiments.util import generate_base_command, generate_run_commands

import experiments.data_gen.generate_data_argus
import argparse
import numpy as np

def main(args):
    rds = np.random.RandomState(args.seed)

    step_sizes = np.exp(np.arange(-5, -0.8, step=0.18))

    command_list = []
    for step_size in step_sizes:

        for j in range(args.num_seeds_per_haparam):
            seed = rds.randint(0, 10**6)
            cmd = generate_base_command(experiments.data_gen.generate_data_argus,
                                        flags={'seed': seed,
                                               'step_size': step_size,
                                               'num_samples': args.num_samples_per_run})
            command_list.append(cmd)

    # submit jobs
    generate_run_commands(command_list, num_cpus=args.num_cpus, mode='local_async', promt=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta-BO run')


    parser.add_argument('--num_cpus', type=int, default=72, help='number of cpus to use')

    parser.add_argument('--seed', type=int, default=382, help='random number generator seed')
    parser.add_argument('--num_seeds_per_haparam', type=int, default=2)
    parser.add_argument('--num_samples_per_run', type=int, default=200)

    args = parser.parse_args()
    main(args)


