from matplotlib import pyplot as plt
import numpy as np
import time

from meta_bo.environment import BraninEnvironment, MixtureEnvironment, ArgusSimEnvironment
from meta_bo.algorithms.acquisition import UCB
from meta_bo.models.vanilla_gp import GPRegressionVanilla

def main():
    env = ArgusSimEnvironment()
    model = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats,
                                normalize_data=True)
    algo = UCB(model, env.domain, beta=2.0)
    evals = []


    for i in range(100):
        x = algo.next()
        t = time.time()
        evaluation = env.evaluate(x)
        eval_time = time.time() - t
        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        algo.add_data(evaluation['x'], evaluation['y'])

        print('step %i | x: %s | t_settle %.6f | evaluation time: %.2f sec'%(evaluation['t'],
                                                                                str(evaluation['x']),
                                                                                evaluation['y'],
                                                                                eval_time))


    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}

    best_eval = np.minimum.accumulate(evals_stacked['y'], axis=-1)

    fig, axes = plt.subplots(ncols=2)

    axes[0].plot(evals_stacked['y'])
    axes[0].set_ylabel('T_settle')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('t')

    axes[1].plot(best_eval)
    axes[1].set_ylabel('T_settle min')
    axes[1].set_xlabel('t')
    fig.show()


if __name__ == '__main__':
    main()