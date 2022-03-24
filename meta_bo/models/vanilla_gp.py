import gpytorch
import numpy as np
import torch

from meta_bo.models.models import LearnedGPRegressionModel, AffineTransformedDistribution
from meta_bo.models.abstract import RegressionModel
from typing import Optional, Dict
from config import device


class GPRegressionVanilla(RegressionModel):

    """
    A Vanilla GP with a Squared-Exponential (SE) Kernel
    """

    def __init__(self, input_dim: int, kernel_variance: float = 2.0, kernel_lengthscale: float = 0.2,
                 likelihood_std: float = 0.05, normalize_data: bool = True, normalization_stats: Optional[Dict] = None,
                 random_state: Optional[np.random.RandomState] = None):
        """ Initializes the GP

        Args:
            input_dim: number of dimensions of x
            kernel_variance: output scale / variance of the SE kernel
            kernel_lengthscale: lengthscale of the SE kernel
            likelihood_std: likelihood std
            normalize_data: whether to standardize the data and work in the standardized data space
            normalization_stats: (optional) dict of normalization stats to use for the standardization
            random_state: (optional) random number generator object
        """

        super().__init__(normalize_data=normalize_data, random_state=random_state)

        """  ------ Setup model ------ """
        assert input_dim > 0
        self.input_dim = input_dim

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)).to(device)
        self.covar_module.outputscale = kernel_variance
        self.covar_module.base_kernel.lengthscale = kernel_lengthscale

        self.mean_module = gpytorch.means.ZeroMean().to(device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.likelihood.noise = likelihood_std

        """ ------- normalization stats & data setup ------- """
        self._set_normalization_stats(normalization_stats)
        self.reset_to_prior()

    def _reset_posterior(self):
        x_context = torch.from_numpy(self.X_data)
        y_context = torch.from_numpy(self.y_data)
        self.gp = LearnedGPRegressionModel(x_context, y_context, self.likelihood,
                                      learned_kernel=None, learned_mean=None,
                                      covar_module=self.covar_module, mean_module=self.mean_module)
        self.gp.eval()
        self.likelihood.eval()

    def _prior(self, x):
        mean_x = self.mean_module(x).squeeze()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def reset_to_prior(self):
        """Clears the training data that was added via add_data(X, y) and resets the posterior to the prior."""
        self._reset_data()
        self.gp = lambda x: self._prior(x)

    def predict(self, test_x: np.ndarray, return_density: bool = False, include_obs_noise: bool = True, **kwargs):
        """
        Given the training data that was provided via the add_data(X, y) method, performs posterior predictions
        for test_x.

        Args:
            test_x (np.ndarray): the points for which to compute predictions
            return_density (bool): whether to return a torch distribution object or
                                   a tuple of (posterior_mean, posterior_std)
            include_obs_noise (bool): whether to include the likelihood std in the posterior predictions.
                                      If yes, the predictions correspond to p(y|x, D) otherwise p(f|x, D)

         Returns:
             Depending on return_density, either a torch distribution or tuple of ndarrays
             (posterior_mean, posterior_std)
        """
        if test_x.ndim == 1:
            test_x = np.expand_dims(test_x, axis=-1)

        with torch.no_grad():
            test_x_normalized = self._normalize_data(test_x)
            test_x_tensor = torch.from_numpy(test_x_normalized).to(device)

            pred_dist = self.gp(test_x_tensor)
            if include_obs_noise:
                pred_dist = self.likelihood(pred_dist)
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)
            if return_density:
                return pred_dist_transformed
            else:
                pred_mean = pred_dist_transformed.mean.cpu().numpy()
                pred_std = pred_dist_transformed.stddev.cpu().numpy()
                return pred_mean, pred_std

    def state_dict(self):
        state_dict = {
            'model': self.gp.state_dict(),
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.gp.load_state_dict(state_dict['model'])

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n_train_samples = 20
    n_test_samples = 200

    torch.manual_seed(25)
    x_data = torch.normal(mean=-1, std=2.0, size=(n_train_samples + n_test_samples, 1))
    W = torch.tensor([[0.6]])
    b = torch.tensor([-1])
    y_data = x_data.matmul(W.T) + torch.sin((0.6 * x_data)**2) + b + torch.normal(mean=0.0, std=0.1, size=(n_train_samples + n_test_samples, 1))
    y_data = torch.reshape(y_data, (-1,))

    x_data_train, x_data_test = x_data[:n_train_samples].numpy(), x_data[n_train_samples:].numpy()
    y_data_train, y_data_test = y_data[:n_train_samples].numpy(), y_data[n_train_samples:].numpy()

    gp_mll = GPRegressionVanilla(input_dim=x_data.shape[-1], kernel_lengthscale=1.)
    gp_mll.add_data(x_data_train, y_data_train)

    x_plot = np.linspace(6, -6, num=200)
    gp_mll.confidence_intervals(x_plot)

    pred_mean, pred_std = gp_mll.predict(x_plot)
    pred_mean, pred_std = pred_mean.flatten(), pred_std.flatten()

    plt.scatter(x_data_test, y_data_test)
    plt.plot(x_plot, pred_mean)

    #lcb, ucb = pred_mean - pred_std, pred_mean + pred_std
    lcb, ucb = gp_mll.confidence_intervals(x_plot)
    plt.fill_between(x_plot, lcb, ucb, alpha=0.4)
    plt.show()
