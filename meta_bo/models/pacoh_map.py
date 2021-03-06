import torch
import gpytorch
import time
import numpy as np

from meta_bo.models.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution
from meta_bo.models.util import _handle_input_dimensionality, DummyLRScheduler
from meta_bo.models.abstract import RegressionModelMetaLearned
from config import device


class PACOH_MAP_GP(RegressionModelMetaLearned):

    def __init__(self, input_dim, learning_mode='both', weight_decay=0.0, feature_dim=2, num_iter_fit=10000,
                 covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 task_batch_size=5, lr=1e-3, lr_decay=1.0, normalize_data=True,
                 normalization_stats=None, random_state=None):

        super().__init__(normalize_data, random_state)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)

        self.input_dim = input_dim
        self.lr, self.weight_decay, self.feature_dim = lr, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data

        """ Setup prior and likelihood """
        self._setup_gp_prior(mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-3)).to(device)
        self.shared_parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr})
        self._setup_optimizer(lr, lr_decay)

        """ ------- normalization stats & data setup  ------- """
        self._normalization_stats = normalization_stats
        self.reset_to_prior()

        self.fitted = False

    def meta_fit(self, meta_train_tuples, meta_valid_tuples=None, verbose=True, log_period=500, n_iter=None):

        assert (meta_valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in meta_valid_tuples]))
        self.likelihood.train()

        task_dicts = self._prepare_meta_train_tasks(meta_train_tuples)

        t = time.time()
        cum_loss = 0.0
        n_iter = self.num_iter_fit if n_iter is None else n_iter

        for itr in range(1, n_iter + 1):
            # actual meta-training step
            task_dict_batch = self._rds.choice(task_dicts, size=self.task_batch_size)
            loss = self._step(task_dict_batch)
            cum_loss += loss

            # print training stats stats
            if itr == 1 or itr % log_period == 0:
                duration = time.time() - t
                avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                cum_loss = 0.0
                t = time.time()

                message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, avg_loss, duration)

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

    def meta_predict(self, context_x, context_y, test_x, return_density=False):
        """
        Performs posterior inference (target training) with (context_x, context_y) as training data and then
        computes the predictive distribution of the targets p(y|test_x, test_context_x, context_y) in the test points

        Args:
            context_x: (ndarray) context input data for which to compute the posterior
            context_y: (ndarray) context targets for which to compute the posterior
            test_x: (ndarray) query input data of shape (n_samples, ndim_x)
            return_density: (bool) whether to return result as mean and std ndarray or as MultivariateNormal pytorch object

        Returns:
            (pred_mean, pred_std) predicted mean and standard deviation corresponding to p(t|test_x, test_context_x, context_y)
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

        task_dicts = [self._dataset_to_task_dict(x, y) for x, y in meta_train_tuples]
        return task_dicts

    def _dataset_to_task_dict(self, x, y):
        # a) prepare data
        x_tensor, y_tensor = self._prepare_data_per_task(x, y)
        task_dict = {'train_x': x_tensor, 'train_y': y_tensor}

        # b) prepare model
        task_dict['model'] = LearnedGPRegressionModel(task_dict['train_x'], task_dict['train_y'], self.likelihood,
                                                      learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                                      covar_module=self.covar_module, mean_module=self.mean_module)
        task_dict['mll_fn'] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict['model'])
        return task_dict

    def _step(self, task_dicts):
        assert len(task_dicts) > 0
        loss = 0.0
        self.optimizer.zero_grad()

        for task_dict in task_dicts:
            output = task_dict['model'](task_dict['train_x'])
            mll = task_dict['mll_fn'](output, task_dict['train_y'])
            loss -= mll

        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

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

    meta_env = RandomMixtureMetaEnv(random_state=np.random.RandomState(29))
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

    weight_decay = 0.01
    gp_model = PACOH_MAP_GP(input_dim=meta_env.domain.d, num_iter_fit=20000,
                            weight_decay=weight_decay, task_batch_size=2,
                            covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                            kernel_nn_layers=NN_LAYERS)
    itrs = 0
    print("---- weight-decay =  %.4f ----"%weight_decay)
    for i in range(10):
        gp_model.meta_fit(meta_train_data, meta_valid_tuples=meta_test_data, log_period=1000, n_iter=500)
        itrs += 500

        x_plot = np.linspace(meta_env.domain.l, meta_env.domain.u, num=150)
        x_context, t_context, x_test, y_test = meta_test_data[0]
        pred_mean, pred_std = gp_model.meta_predict(x_context, t_context, x_plot)
        ucb, lcb = (pred_mean + 2 * pred_std).flatten(), (pred_mean - 2 * pred_std).flatten()

        plt.scatter(x_test, y_test)
        plt.scatter(x_context, t_context)

        plt.plot(x_plot, pred_mean)
        plt.fill_between(x_plot.flatten(), lcb, ucb, alpha=0.2)
        plt.title('GPR meta mll (weight-decay =  %.4f) itrs = %i' % (weight_decay, itrs))
        plt.show()