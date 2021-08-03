from matplotlib import pyplot as plt
import numpy as np
from pprint import pprint

from meta_bo.environment import BraninEnvironment, MixtureEnvironment
from meta_bo.algorithms.acquisition import GooseUCB
from meta_bo.models.vanilla_gp import GPRegressionVanilla

def main():
    env = MixtureEnvironment()
    model = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats,
                                normalize_data=True)

    model_constr = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats_constr,
                                normalize_data=True)

    x_plot = np.expand_dims(np.linspace(-10, 10, 200), axis=-1)
    f = env.f(x_plot)
    q = env.q_constraint(x_plot)
    plt.plot(x_plot, f, label='f(x)')
    plt.plot(x_plot, q, label='q(x)')
    plt.plot(x_plot, np.zeros(x_plot.shape[0]), label='0', linestyle='--')
    plt.legend()
    plt.show()

    algo = GooseUCB(model, model_constr, env.domain, beta=2.0)
    evals = []


    for t in range(50):
        x = algo.next()
        x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)

        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
        pprint(evaluation)
        algo.add_data(evaluation['x'], evaluation['y'], evaluation['q'])

        if t % 5 == 0 and t > 1:
            x_plot = np.expand_dims(np.linspace(-10, 10, 200), axis=-1)
            pred_mean, pred_std = model.predict(x_plot)
            pred_mean_constr, pred_std_constr = model_constr.predict(x_plot)

            fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
            axes[0].plot(x_plot, pred_mean, label='pred fun')
            axes[0].fill_between(np.squeeze(x_plot), pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.25)
            axes[0].plot(x_plot, env.f(x_plot), label='target fun')
            axes[0].scatter(evals_stacked['x'], evals_stacked['y'], label='evaluations')
            axes[0].legend()

            axes[1].plot(x_plot, pred_mean_constr, label='pred constr.')
            axes[1].fill_between(np.squeeze(x_plot), pred_mean_constr - 2 * pred_std_constr,
                                 pred_mean_constr + 2 * pred_std_constr, alpha=0.25)
            axes[1].plot(x_plot, env.q_constraint(x_plot), label='true constr.')
            axes[1].plot(x_plot, np.zeros(x_plot.shape[0]), label='safety threshold', linestyle='--')
            axes[1].scatter(evals_stacked['x'], evals_stacked['q'], label='constr. evals')
            axes[1].legend()

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