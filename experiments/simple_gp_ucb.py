from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np

from meta_bo.environment import BraninEnvironment, MixtureEnvironment
from meta_bo.algorithms.acquisition import UCB
from meta_bo.models.vanilla_gp import GPRegressionVanilla

def main():
    rds = np.random.RandomState(134)
    env = BraninEnvironment(random_state=rds)
    model = GPRegressionVanilla(input_dim=env.domain.d, normalization_stats=env.normalization_stats,
                                normalize_data=True, kernel_lengthscale=0.3, random_state=rds)
    algo = UCB(model, env.domain, beta=2.0, random_state=rds)
    evals = []

    for t in range(100):
        x = algo.next()
        x_bp = algo.best_predicted()
        evaluation = env.evaluate(x, x_bp=x_bp)
        evals.append(evaluation)
        evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}

        algo.add_data(evaluation['x'], evaluation['y'])

        if t % 20 == 0 and t > 1:
            if env.domain.d == 1:
                x_plot = np.expand_dims(np.linspace(-10, 10, 200), axis=-1)
                pred_mean, pred_std = model.predict(x_plot)
                plt.plot(x_plot, pred_mean)
                plt.fill_between(np.squeeze(x_plot), pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.25)
                plt.scatter(evals_stacked['x'], evals_stacked['y'])
                plt.show()
            elif env.domain.d == 2:
                fig, ax = plt.subplots(1, 1)
                x1, x2 = np.meshgrid(np.linspace(env.domain.l[0], env.domain.u[0], 100),
                                     np.linspace(env.domain.l[1], env.domain.u[1], 100))
                f = env.f(np.stack([x1.flatten(), x2.flatten()], axis=-1)).reshape(x1.shape)
                contour_f = ax.contour(x1, x2, f, origin='lower', locator=ticker.LogLocator())
                ax.scatter(evals_stacked['x'][:, 0], evals_stacked['x'][:, 1])
                fig.colorbar(contour_f)
                fig.show()
                print(f'iter {t}')


    """ plot regret """
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}
    regret = evals_stacked['y_exact'] - evals_stacked['y_min']
    regret_bp = evals_stacked['y_exact_bp'] - evals_stacked['y_min']

    simple_regret = np.minimum.accumulate(regret, axis=-1)
    cum_regret = np.cumsum(regret, axis=-1)
    cum_regret_bp = np.cumsum(regret_bp, axis=-1)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4))

    axes[0].plot(simple_regret)
    axes[0].set_ylabel('simple regret')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('t')

    axes[1].plot(cum_regret_bp)
    axes[1].set_ylabel('cumulative inference regret')
    axes[1].set_xlabel('t')
    fig.tight_layout()
    fig.show()

if __name__ == '__main__':
    main()