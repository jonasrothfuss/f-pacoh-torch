import torch
import gpytorch
import time
import math
import numpy as np
from torch.distributions import Uniform, MultivariateNormal, kl_divergence

from meta_bo.models.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution, SEKernelLight
from meta_bo.models.util import _handle_input_dimensionality, DummyLRScheduler
from meta_bo.models.abstract import RegressionModelMetaLearned
from meta_bo.domain import Domain, ContinuousDomain, DiscreteDomain
from config import device

from typing import List, Optional, Dict, Tuple


class FPACOH_MAP_GP(RegressionModelMetaLearned):
    """
    Implementation of the F-PACOH-GP algorithm with a GP (SE kernel) as a hyper-prior.
    Link to the paper 'Meta-Learning Reliable Priors in the Function Space': https://arxiv.org/abs/2106.03195.
    """

    def __init__(self, domain: Domain, learning_mode: str = 'both', weight_decay: float = 0.0, feature_dim: int = 2,
                 num_iter_fit: int = 10000, covar_module: str = 'NN', mean_module: str = 'NN',
                 mean_nn_layers: List[int] = (32, 32), kernel_nn_layers: List[int] = (32, 32),
                 prior_lengthscale: float = 0.2, prior_outputscale: float = 2.0, prior_kernel_noise: float = 1e-3,
                 train_data_in_kl: bool = True, num_samples_kl: int = 20, task_batch_size: int = 10, lr: float = 1e-3,
                 lr_decay: float = 1.0, prior_factor: float = 0.1, normalize_data: bool = True,
                 normalization_stats: Optional[Dict] = None, random_state: Optional[np.random.RandomState] = None):
        """

        Args:
            domain: data domain
            learning_mode: one of ['both', 'mean', 'kernel', 'vanilla']. Indicates whether to train both the kernel
                           and the mean function or only one of them.
            weight_decay: amount of weight decay to use for meta-training
            feature_dim: dimensionality of the latent feature space of the kernel map
            num_iter_fit: number of meta-training iterations
            covar_module: one of ['NN', 'SE']. how to parametrize the kernel function.
            mean_module: one of ['NN', 'constant', 'zero]. how to parametrize the mean function.
            mean_nn_layers: hidden layer sizes of the mean NN
            kernel_nn_layers: hidden layer sizes of the kernel feature map NN
            prior_lengthscale: lengthscale of the SE hyper-prior
            prior_outputscale: outputscale / variance of the SE hyper-prior
            prior_kernel_noise: magnitude of identity matrix to add to hyper-prior kernel matrix
                                for numerical stability
            train_data_in_kl: whether to include the meta-train data in the computation of the functional KL or only
                              to use the uniformly sampled measurement sets
            num_samples_kl: size of measurement set that is used to compute the functional kl
            task_batch_size: task batch size
            lr: Adam learning rate
            lr_decay: multiplicative learning rate decay factor which is applied every 1000 iter to the lr
            normalize_data: whether to standardize the data and work in the standardized data space
            prior_factor: multiplicative factor for weighting the function KL relative to the mlls
            normalization_stats: (optional) dict of normalization stats to use for the standardization
            random_state: (optional) random number generator object
        """
        super().__init__(normalize_data, random_state)

        assert isinstance(domain, ContinuousDomain) or isinstance(domain, DiscreteDomain)
        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)

        self.domain = domain
        self.input_dim = domain.d
        self.lr, self.weight_decay, self.feature_dim = lr, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data

        """ Setup prior and likelihood """
        self._setup_gp_prior(mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-3)).to(device)
        self.shared_parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr})
        self._setup_optimizer(lr, lr_decay)

        """ domain support dist & prior kernel """
        self.prior_factor = prior_factor
        self.domain_dist = Uniform(low=torch.from_numpy(domain.l).float(), high=torch.from_numpy(domain.u).float())
        lengthscale = prior_lengthscale * torch.ones((1, self.input_dim))
        self.prior_covar_module = SEKernelLight(lengthscale, output_scale=torch.tensor(prior_outputscale))
        self.prior_kernel_noise = prior_kernel_noise
        self.train_data_in_kl = train_data_in_kl
        self.num_samples_kl = num_samples_kl

        """ ------- normalization stats & data setup  ------- """
        self._normalization_stats = normalization_stats
        self.reset_to_prior()

        self.fitted = False
        self._meta_train_iter = 0

    def meta_fit(self, meta_train_tuples: List[Tuple[np.ndarray, np.ndarray]],
                 meta_valid_tuples: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = None,
                 verbose: bool = True, log_period: int = 500, n_iter: Optional[int] = None):
        """ Performs the meta-training of the GP prior with the F-PACOH-MAP algorithm.

        Args:
            meta_train_tuples: lest of meta-train tuples, i.e. [(x_train_1, y_train_1), .... ]
            meta_valid_tuples: list of meta-valid tuples, i.e. [(test_context_x_1, test_context_t_1,
                               test_x_1, test_t_1), ...]
            verbose: whether to print training statistics
            log_period: number of iterations after which to log
            n_iter: number of iterations, overwrites the num_iter_fit variable which is set in __init__

        Returns: (float) loss at last iteration

        """
        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        self.likelihood.train()

        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        for itr in range(1, n_iter + 1):
            self._meta_train_iter += 1

            # actual meta-training step
            task_dict_batch = self._rds.choice(task_dicts, size=self.task_batch_size)
            loss = self._step(task_dict_batch, n_tasks=len(task_dicts))
            cum_loss += loss

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                cum_loss = 0.0
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (self._meta_train_iter, self.num_iter_fit,
                                                                       avg_loss, duration)

                # if validation data is provided  -> compute the valid log-likelihood
                if meta_valid_tuples is not None:
                    self.likelihood.eval()
                    valid_ll, valid_rmse, calibr_err, calibr_err_chi2 = self.eval_datasets(meta_valid_tuples)
                    self.likelihood.train()
                    message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_ll, valid_rmse, calibr_err)

                if verbose:
                    print(message)

        self.fitted = True

        # set gpytorch modules to eval mode and set gp to meta-learned gp prior
        assert self.X_data.shape[0] == 0 and self.y_data.shape[0] == 0, "Data for posterior inference can be passed " \
                                                                        "only after the meta-training"
        for task_dict in task_dicts:
            task_dict['model'].eval()
        self.likelihood.eval()
        self.reset_to_prior()
        return loss

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
            test_x_tensor = torch.from_numpy(test_x_normalized).float().to(device)

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

    def meta_predict(self, context_x: np.ndarray, context_y: np.ndarray, test_x: np.ndarray,
                     return_density: bool = False):
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            Depending on return_density either (pred_mean, pred_std) posterior mean and standard deviation
            corresponding to p(test_y|test_x, test_context_x, context_y), or a corresponding torch density object
        """

        context_x, context_y = _handle_input_dimensionality(context_x, context_y)
        test_x = _handle_input_dimensionality(test_x)
        assert test_x.shape[1] == context_x.shape[1]

        # normalize data and convert to tensor
        context_x, context_y = self._prepare_data_per_task(context_x, context_y)

        test_x = self._normalize_data(X=test_x, Y=None)
        test_x = torch.from_numpy(test_x).float().to(device)

        with torch.no_grad():
            # compute posterior given the context data
            gp_model = LearnedGPRegressionModel(context_x, context_y, self.likelihood,
                                                learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                covar_module=self.covar_module, mean_module=self.mean_module)
            gp_model.eval()
            self.likelihood.eval()
            pred_dist = self.likelihood(gp_model(test_x))
            pred_dist_transformed = AffineTransformedDistribution(pred_dist, normalization_mean=self.y_mean,
                                                                  normalization_std=self.y_std)

        if return_density:
            return pred_dist_transformed
        else:
            pred_mean = pred_dist_transformed.mean
            pred_std = pred_dist_transformed.stddev
            return pred_mean.cpu().numpy(), pred_std.cpu().numpy()

    def reset_to_prior(self):
        """Clears the training data that was added via add_data(X, y) and resets the posterior to the prior."""
        self._reset_data()
        self.gp = lambda x: self._prior(x)

    def _reset_posterior(self):
        x_context = torch.from_numpy(self.X_data).float().to(device)
        y_context = torch.from_numpy(self.y_data).float().to(device)
        self.gp = LearnedGPRegressionModel(x_context, y_context, self.likelihood,
                                      learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                      covar_module=self.covar_module, mean_module=self.mean_module)
        self.gp.eval()
        self.likelihood.eval()

    def state_dict(self):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'model': self.task_dicts[0]['model'].state_dict()
        }
        for task_dict in self.task_dicts:
            for key, tensor in task_dict['model'].state_dict().items():
                assert torch.all(state_dict['model'][key] == tensor).item()
        return state_dict

    def load_state_dict(self, state_dict):
        for task_dict in self.task_dicts:
            task_dict['model'].load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _prior(self, x):
        if self.nn_kernel_map is not None:
            projected_x = self.nn_kernel_map(x)
        else:
            projected_x = x

            # feed through mean module
        if self.nn_mean_fn is not None:
            mean_x = self.nn_mean_fn(x).squeeze()
        else:
            mean_x = self.mean_module(projected_x).squeeze()

        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _prepare_meta_train_tasks(self, meta_train_tuples):
        self._check_meta_data_shapes(meta_train_tuples)

        if self._normalization_stats is None:
            self._compute_meta_normalization_stats(meta_train_tuples)
        else:
            self._set_normalization_stats(self._normalization_stats)

        self.x_mean_torch = torch.from_numpy(self.x_mean).float()
        self.x_std_torch = torch.from_numpy(self.x_std).float()
        task_dicts = [self._dataset_to_task_dict(x, y) for x, y in meta_train_tuples]
        return task_dicts

    def _dataset_to_task_dict(self, x, y):
        # a) prepare data
        x_tensor, y_tensor = self._prepare_data_per_task(x, y)
        task_dict = {'x_train': x_tensor, 'y_train': y_tensor}

        # b) prepare model
        task_dict['model'] = LearnedGPRegressionModel(task_dict['x_train'], task_dict['y_train'], self.likelihood,
                                                      learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                      covar_module=self.covar_module, mean_module=self.mean_module)
        task_dict['mll_fn'] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict['model'])
        return task_dict

    def _step(self, task_dict_batch, n_tasks):
        assert len(task_dict_batch) > 0
        loss = 0.0
        self.optimizer.zero_grad()

        for task_dict in task_dict_batch:
            # mll term
            output = task_dict['model'](task_dict['x_train'])
            mll = task_dict['mll_fn'](output, task_dict['y_train'])

            # kl term
            kl = self._f_kl(task_dict)

            #  terms for pre-factors
            n = n_tasks
            m = task_dict['x_train'].shape[0]

            # loss for this batch
            loss += - mll / (self.task_batch_size * m) + \
                    self.prior_factor * (1 / math.sqrt(n) + 1 / (n * m)) * kl / self.task_batch_size

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def _sample_measurement_set(self, x_train):
        if self.train_data_in_kl:
            n_train_x = min(x_train.shape[0], self.num_samples_kl // 2)
            n_rand_x = self.num_samples_kl - n_train_x
            idx_rand = np.random.choice(x_train.shape[0], n_train_x)
            x_domain_dist = self._normalize_x_torch(self.domain_dist.sample((n_rand_x,)))
            x_kl = torch.cat([x_train[idx_rand], x_domain_dist], dim=0)
        else:
            x_kl = self._normalize_x_torch(self.domain_dist.sample((self.num_samples_kl,)))
        assert x_kl.shape == (self.num_samples_kl, self.input_dim)
        return x_kl

    def _normalize_x_torch(self, X):
        assert hasattr(self, "x_mean_torch") and hasattr(self, "x_std_torch"), (
            "requires computing normalization stats beforehand")
        x_normalized = (X - self.x_mean_torch[None, :]) / self.x_std_torch[None, :]
        return x_normalized

    def _f_kl(self, task_dict):
        with gpytorch.settings.debug(False):

            # sample / construc measurement set
            x_kl = self._sample_measurement_set(task_dict['x_train'])

            # functional KL
            dist_f_posterior = task_dict['model'](x_kl)
            K_prior = torch.reshape(self.prior_covar_module(x_kl).evaluate(), (x_kl.shape[0], x_kl.shape[0]))

            inject_noise_std = self.prior_kernel_noise
            error_counter = 0
            while error_counter < 5:
                try:
                    dist_f_prior = MultivariateNormal(torch.zeros(x_kl.shape[0]), K_prior + inject_noise_std * torch.eye(x_kl.shape[0]))
                    return kl_divergence(dist_f_posterior, dist_f_prior)
                except RuntimeError as e:
                    import warnings
                    inject_noise_std = 2 * inject_noise_std
                    error_counter += 1
                    warnings.warn('encoundered numerical error in computation of KL: %s '
                                  '--- Doubling inject_noise_std to %.4f and trying again' % (str(e), inject_noise_std))
            raise RuntimeError('Not able to compute KL')

    def _setup_gp_prior(self, mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers):

        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            self.nn_kernel_map = NeuralNetwork(input_dim=self.input_dim, output_dim=feature_dim,
                                          layer_sizes=kernel_nn_layers).to(device)
            self.shared_parameters.append(
                {'params': self.nn_kernel_map.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim)).to(device)
        else:
            self.nn_kernel_map = None

        if covar_module == 'SE':
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=self.input_dim)).to(device)
        elif isinstance(covar_module, gpytorch.kernels.Kernel):
            self.covar_module = covar_module.to(device)

        # b) determine mean map & module

        if mean_module == 'NN':
            assert learning_mode in ['learn_mean', 'both'], 'neural network parameters must be learned'
            self.nn_mean_fn = NeuralNetwork(input_dim=self.input_dim, output_dim=1, layer_sizes=mean_nn_layers).to(device)
            self.shared_parameters.append(
                {'params': self.nn_mean_fn.parameters(), 'lr': self.lr, 'weight_decay': self.weight_decay})
            self.mean_module = None
        else:
            self.nn_mean_fn = None

        if mean_module == 'constant':
            self.mean_module = gpytorch.means.ConstantMean().to(device)
        elif mean_module == 'zero':
            self.mean_module = gpytorch.means.ZeroMean().to(device)
        elif isinstance(mean_module, gpytorch.means.Mean):
            self.mean_module = mean_module.to(device)

        # c) add parameters of covar and mean module if desired

        if learning_mode in ["learn_kernel", "both"]:
            self.shared_parameters.append({'params': self.covar_module.hyperparameters(), 'lr': self.lr})

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append({'params': self.mean_module.hyperparameters(), 'lr': self.lr})

    def _setup_optimizer(self, lr, lr_decay):
        self.optimizer = torch.optim.AdamW(self.shared_parameters, lr=lr, weight_decay=self.weight_decay)
        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)


if __name__ == "__main__":
    from meta_bo.meta_environment import RandomMixtureMetaEnv

    meta_env = RandomMixtureMetaEnv(random_state=np.random.RandomState(26))
    meta_train_data = meta_env.generate_uniform_meta_train_data(num_tasks=20, num_points_per_task=10)
    meta_test_data = meta_env.generate_uniform_meta_valid_data(num_tasks=50, num_points_context=10, num_points_test=160)

    NN_LAYERS = (32, 32)

    plot = True
    from matplotlib import pyplot as plt

    if plot:
        for x_train, y_train in meta_train_data:
            plt.scatter(x_train, y_train)
        plt.title('sample from the GP prior')
        plt.show()

    """ 2) Classical mean learning based on mll """

    print('\n ---- GPR mll meta-learning ---- ')

    torch.set_num_threads(2)

    prior_factor = 1e0
    gp_model = FPACOH_MAP_GP(domain=meta_env.domain, num_iter_fit=20000, weight_decay=1e-4, prior_factor=prior_factor,
                             task_batch_size=2, covar_module='NN', mean_module='NN',
                             mean_nn_layers=NN_LAYERS, kernel_nn_layers=NN_LAYERS)
    itrs = 0
    for i in range(10):
        gp_model.meta_fit(meta_train_data, meta_valid_tuples=meta_test_data, log_period=250, n_iter=1000)
        itrs += 1000

        x_plot = np.linspace(meta_env.domain.l, meta_env.domain.u, num=150)
        x_context, t_context, x_test, y_test = meta_test_data[0]
        pred_mean, pred_std = gp_model.meta_predict(x_context, t_context, x_plot)
        ucb, lcb = (pred_mean + 2 * pred_std).flatten(), (pred_mean - 2 * pred_std).flatten()

        plt.scatter(x_test, y_test)
        plt.scatter(x_context, t_context)

        plt.plot(x_plot, pred_mean)
        plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2)
        plt.title('GPR meta mll (prior_factor =  %.4f) itrs = %i' % (prior_factor, itrs))
        plt.show()