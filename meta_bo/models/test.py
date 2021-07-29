from meta_bo.models.random_gp import VectorizedGP

from meta_bo.models.models import LearnedGPRegressionModel, ConstantMeanLight, SEKernelLight, GaussianLikelihoodLight, \
        VectorizedModel, CatDist, NeuralNetworkVectorized

import numpy as np
import torch
import gpytorch

x = np.random.uniform(-3, 3, size=(50, 1))
f = lambda x : np.sin(x)
y = f(x) + 0.1 * np.random.normal(size=x.shape)

from matplotlib import pyplot as plt
plt.scatter(x, y)


x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float().flatten()
x_plot = torch.from_numpy(np.expand_dims(np.linspace(-4, 4, 100), axis=-1)).float()

# mean_module = ConstantMeanLight(constant=torch.zeros(1))
# covar_module = SEKernelLight(lengthscale=torch.ones(1,1))
#likelihood = GaussianLikelihoodLight(noise_var=0.1 * torch.ones(1))


#mean_module = gpytorch.means.ConstantMean()
#covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
# likelihood = gpytorch.likelihoods.GaussianLikelihood()
#
# gp = LearnedGPRegressionModel(x, y, likelihood, mean_module=mean_module, covar_module=covar_module,
#                               learned_mean=None, learned_kernel=None)

input_dim = 1
mean_module = ConstantMeanLight(torch.zeros(2,1))

lengthscale = torch.ones((2, 1, input_dim))
covar_module = SEKernelLight(lengthscale, output_scale=torch.tensor(1.0))

noise = 0.1 * torch.ones((2,1))
likelihood = GaussianLikelihoodLight(noise)

x = x.unsqueeze(0).repeat((2, 1, 1))
y = y.unsqueeze(0).repeat((2, 1))
gp = LearnedGPRegressionModel(x, y, likelihood, mean_module=mean_module, covar_module=covar_module,
                              learned_mean=None, learned_kernel=None)


likelihood.train()
gp.train()

x_query = x_plot.unsqueeze(0).repeat((2, 1, 1))
pred_dist = gp(x_query)

y_pred = pred_dist.mean[0].numpy()
y_std = pred_dist.stddev[0].numpy()

plt.plot(x_plot.numpy(), y_pred)
plt.fill_between(np.squeeze(x_plot), y_pred-y_std, y_pred+y_std, alpha=0.3)
plt.show()