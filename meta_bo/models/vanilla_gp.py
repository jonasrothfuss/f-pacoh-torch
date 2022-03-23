import gpytorch
import numpy as np
import torch

from meta_bo.models.models import LearnedGPRegressionModel, AffineTransformedDistribution
from meta_bo.models.abstract import RegressionModel
from config import device


class GPRegressionVanilla(RegressionModel):

    def __init__(self, input_dim, kernel_variance=2.0, kernel_lengthscale=0.2, likelihood_std=0.05,
                 normalize_data=True, normalization_stats=None, random_state=None):

        super().__init__(normalize_data=normalize_data, random_state=random_state)

        """  ------ Setup model ------ """
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
        self._reset_data()
        self.gp = lambda x: self._prior(x)

    def predict(self, test_x: np.ndarray, return_density: bool = False, include_obs_noise: bool = True, **kwargs):
        """
        computes the predictive distribution of the targets p(t|test_x, train_x, train_y)

        Args:
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density (bool) whether to return a density object or

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(y_test|X_test, X_train, y_train)
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

    def predict_mean_std(self, test_x):
        return self.predict(test_x, return_density=False)

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
