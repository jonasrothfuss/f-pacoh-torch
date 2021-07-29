from matplotlib import pyplot as plt
import numpy as np
from absl import logging

from meta_bo.meta_environment import RandomBraninMetaEnv, RandomMixtureMetaEnv
from meta_bo.algorithms.acquisition import UCB
from meta_bo.models.pacoh_map import PACOH_MAP_GP
from meta_bo.models.f_pacoh_map import FPACOH_MAP_GP

import time
t = time.time()

meta_env = RandomMixtureMetaEnv()
meta_train_data = meta_env.generate_uniform_meta_train_data(20, 20)
meta_valid_data = meta_env.generate_uniform_meta_valid_data(10, 10, 100)
print('time to generate data: %.2f sec'%(time.time() - t))

# from matplotlib import pyplot as plt
# for x, y in meta_train_data:
#     plt.scatter(x, y)
# plt.show()

logging.set_verbosity(logging.INFO)

MODEL = 'FPACOH'

if MODEL == 'PACOH':
    model = PACOH_MAP_GP(input_dim=meta_env.domain.d, normalization_stats=meta_env.normalization_stats,
                         normalize_data=True, num_iter_fit=100, weight_decay=0.01, lr=0.02,
                         covar_module='SE', mean_module='constant')
elif MODEL == 'FPACOH':
    model = FPACOH_MAP_GP(domain=meta_env.domain, normalization_stats=meta_env.normalization_stats,
                          num_iter_fit=5000, weight_decay=0.0001, prior_factor=0.5)
else:
    raise NotImplementedError

model.meta_fit(meta_train_data, meta_valid_tuples=meta_valid_data, log_period=100)


algo = UCB(model, meta_env.domain, beta=2.0)
evals = []

env = meta_env.sample_env()

for t in range(50):
    x = algo.next()
    x_bp = algo.best_predicted()
    evaluation = env.evaluate(x, x_bp=x_bp)
    evals.append(evaluation)
    evals_stacked = {k: np.array([dic[k] for dic in evals]) for k in evals[0]}

    algo.add_data(evaluation['x'], evaluation['y'])

    if t % 5 == 0 and t > 1:
        x_plot = np.expand_dims(np.linspace(-10, 10, 200), axis=-1)
        pred_mean, pred_std = model.predict(x_plot)
        plt.plot(x_plot, pred_mean, label='pred_mean')
        plt.fill_between(np.squeeze(x_plot), pred_mean - 2 * pred_std, pred_mean + 2 * pred_std, alpha=0.25)
        plt.scatter(evals_stacked['x'], evals_stacked['y'], label='evaluations')
        plt.plot(x_plot, env.f(x_plot.flatten()), label='target fun')
        plt.legend()
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


