from matplotlib import pyplot as plt, ticker
import numpy as np
import time

from meta_bo.meta_environment import RandomBraninMetaEnv
from meta_bo.environment import BraninEnvironment
from meta_bo.algorithms.acquisition import UCB
from meta_bo.models.f_pacoh_map import FPACOH_MAP_GP



rds = np.random.RandomState(134)

meta_env = RandomBraninMetaEnv(random_state=rds)
meta_train_data = meta_env.generate_uniform_meta_train_data(100, 50)
meta_valid_data = meta_env.generate_uniform_meta_valid_data(40, 20, 100)

NN_LAYERS = [32, 32, 32]
model = FPACOH_MAP_GP(domain=meta_env.domain, normalization_stats=meta_env.normalization_stats,
                      num_iter_fit=10000, weight_decay=1e-4, prior_factor=0.045, num_samples_kl=40,
                      mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS, prior_lengthscale=0.3,
                      task_batch_size=10,  random_state=rds)

model.meta_fit(meta_train_data, meta_valid_tuples=meta_valid_data, log_period=1000)


algo = UCB(model, meta_env.domain, beta=2.0, random_state=rds)
evals = []

env = BraninEnvironment(random_state=rds)

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
            x1, x2 = np.meshgrid(np.linspace(env.domain.l[0], env.domain.u[0], 100),
                                 np.linspace(env.domain.l[1], env.domain.u[1], 100))
            f = env.f(np.stack([x1.flatten(), x2.flatten()], axis=-1)).reshape(x1.shape)
            contour_f = plt.contour(x1, x2, f, origin='lower', locator=ticker.LogLocator())
            plt.scatter(evals_stacked['x'][:, 0], evals_stacked['x'][:, 1])
            plt.colorbar(contour_f)
            plt.show()

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
fig.save_fig()


