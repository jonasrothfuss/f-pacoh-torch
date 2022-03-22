from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint

from meta_bo.environment import BraninEnvironment, MixtureEnvironment
from meta_bo.algorithms.acquisition import GooseUCB
from meta_bo.models.vanilla_gp import GPRegressionVanilla
from meta_bo.domain import DiscreteDomain

def main():
    env = BraninEnvironment()
    model = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats,
                                normalize_data=True)

    model_constr = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats_constr,
                                       normalize_data=True)

    x_plot = np.expand_dims(np.linspace(-10, 10, 200), axis=-1)

    domain = DiscreteDomain.uniform_from_continuous_domain(env.domain)
    algo = GooseUCB(model, model_constr, domain, beta=2.0, x0= 6 * np.ones(env.domain.d))
    evals = []


    for t in range(200):
        x = algo.next()
        x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)

        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        pprint(evaluation)
        algo.add_data(evaluation['x'], evaluation['y'], evaluation['q'])

        if t % 1 == 0 and t > 1:
            fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
            x1, x2 = np.meshgrid(np.arange(env.domain.l[0], env.domain.u[0], 0.05),
                                 np.arange(env.domain.l[1], env.domain.u[1], 0.05))
            f = env.f(np.stack([x1.flatten(), x2.flatten()], axis=-1)).reshape(x1.shape)
            q = env.q_constraint(np.stack([x1.flatten(), x2.flatten()], axis=-1)).reshape(x1.shape)
            contour_f = axes[0].contour(x1, x2, f, origin='lower')
            axes[0].scatter(evals_stacked['x'][:, 0], evals_stacked['x'][:, 1])

            # pessimistic_safe_set, optimistic_safe_set, uncertain_safety_set, informative_expanders = algo.get_safe_sets()
            #xes[1].scatter(uncertain_safety_set[:, 0], uncertain_safety_set[:, 1])
            # axes[1].scatter(pessimistic_safe_set[:, 0], pessimistic_safe_set[:, 1])
            pessimistic_safe_set = algo._is_in_pessimistic_safe_set(np.stack([x1.flatten(), x2.flatten()], axis=-1)).astype(np.float)

            axes[1].pcolormesh(x1, x2, pessimistic_safe_set.reshape(x1.shape), cmap='copper')

            axes[1].contour(x1, x2, q, levels=[0,], origin='lower')
            plt.colorbar(contour_f)
            plt.show()


    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    regret = evals_stacked['y_exact'] - evals_stacked['y_min']
    regret_bp = evals_stacked['y_exact_bp'] - evals_stacked['y_min']

    simple_regret = np.minimum.accumulate(regret, axis=-1)
    cum_regret = np.cumsum(regret, axis=-1)
    cum_regret_bp = np.cumsum(regret_bp, axis=-1)

    fig, axes = plt.subplots(ncols=2)

    axes[0].plot(simple_regret)
    axes[0].set_ylabel('simple regret')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('t')

    axes[1].plot(cum_regret_bp)
    axes[1].set_ylabel('cumulative inference regret')
    axes[1].set_xlabel('t')
    fig.show()


if __name__ == '__main__':
    main()