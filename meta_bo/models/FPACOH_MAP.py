import torch
import gpytorch
import time

import warnings
import math
import numpy as np

from gpytorch.utils.warnings import NumericalWarning

from meta_bo.models.models import LearnedGPRegressionModel, NeuralNetwork, AffineTransformedDistribution, SEKernelLight
from meta_bo.models.util import _handle_input_dimensionality, DummyLRScheduler
from meta_bo.models.abstract import RegressionModelMetaLearned
from config import device

from torch.distributions import Uniform, MultivariateNormal, kl_divergence

class FPACOH_MAP(RegressionModelMetaLearned):

    def __init__(self, meta_train_data, learning_mode='both', lr_params=1e-3, weight_decay=0.0, feature_dim=2,
                 num_iter_fit=10000, covar_module='NN', mean_module='NN', mean_nn_layers=(32, 32), kernel_nn_layers=(32, 32),
                 prior_factor=0.1, domain_l=-4.0, domain_h=4.0, prior_lengthscale=0.2, prior_outputscale=2.0, prior_kernel_noise=1e-3,
                 task_batch_size=4, normalize_data=True, train_data_in_kl=True, num_samples_kl=20,
                 optimizer='Adam', lr_decay=1.0, random_seed=None):
        """
        Meta-Learning GP priors (i.e. mean and kernel function) via PACOH-MAP

        Args:
            meta_train_data: list of tuples of ndarrays[(train_x_1, train_t_1), ..., (train_x_n, train_t_n)]
            learning_mode: (str) specifying which of the GP prior parameters to optimize. Either one of
                    ['learned_mean', 'learned_kernel', 'both', 'vanilla']
            lr_params: (float) learning rate for GP prior parameters
            weight_decay: (float) weight decay multiplier for meta-level regularization
            feature_dim: (int) output dimensionality of NN feature map for kernel function
            num_iter_fit: (int) number of gradient steps for fitting the parameters
            covar_module: (gpytorch.mean.Kernel) optional kernel module, default: RBF kernel
            mean_module: (gpytorch.mean.Mean) optional mean module, default: ZeroMean
            mean_nn_layers: (tuple) hidden layer sizes of mean NN
            kernel_nn_layers: (tuple) hidden layer sizes of kernel NN
            learning_rate: (float) learning rate for AdamW optimizer
            task_batch_size: (int) batch size for meta training, i.e. number of tasks for computing gradients
            optimizer: (str) type of optimizer to use - must be either 'Adam' or 'SGD'
            lr_decay: (str) multiplicative learning rate decay applied every 1000 iterations
            random_seed: (int) seed for pytorch
        """
        super().__init__(normalize_data, random_seed)

        assert learning_mode in ['learn_mean', 'learn_kernel', 'both', 'vanilla']
        assert mean_module in ['NN', 'constant', 'zero'] or isinstance(mean_module, gpytorch.means.Mean)
        assert covar_module in ['NN', 'SE'] or isinstance(covar_module, gpytorch.kernels.Kernel)
        assert optimizer in ['Adam', 'SGD']

        self.lr_params, self.weight_decay, self.feature_dim = lr_params, weight_decay, feature_dim
        self.num_iter_fit, self.task_batch_size, self.normalize_data = num_iter_fit, task_batch_size, normalize_data

        # Check that data all has the same size
        self._check_meta_data_shapes(meta_train_data)
        self._compute_normalization_stats(meta_train_data)

        # Setup components that are shared across tasks
        self._setup_gp_prior(mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.likelihoods.noise_models.GreaterThan(1e-3)).to(device)
        self.shared_parameters.append({'params': self.likelihood.parameters(), 'lr': self.lr_params, 'weight_decay': 0.0})

        # domain support dist & prior kernel
        self.prior_factor = prior_factor
        self.domain_dist = Uniform(low=domain_l*torch.ones(self.input_dim), high=domain_h*torch.ones(self.input_dim))
        lengthscale = prior_lengthscale * torch.ones((1, self.input_dim))
        self.prior_covar_module = SEKernelLight(lengthscale, output_scale=torch.tensor(prior_outputscale))
        self.prior_kernel_noise = prior_kernel_noise
        self.train_data_in_kl = train_data_in_kl
        self.num_samples_kl = num_samples_kl

        # Setup components that are different across tasks
        self.task_dicts = []

        for train_x, train_y in meta_train_data:
            task_dict = {}

            # a) prepare data
            x_tensor, y_tensor = self._prepare_data_per_task(train_x, train_y)
            task_dict['x_train'], task_dict['y_train'] = x_tensor, y_tensor

            # b) prepare model
            task_dict['model'] = LearnedGPRegressionModel(task_dict['x_train'], task_dict['y_train'], self.likelihood,
                                              learned_kernel=self.nn_kernel_map, learned_mean=self.nn_mean_fn,
                                              covar_module=self.covar_module, mean_module=self.mean_module)
            task_dict['mll_fn'] = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, task_dict['model']).to(device)

            self.task_dicts.append(task_dict)

        # setup GP hyper-prior
        self.hyper_prior_mean_module = gpytorch.means.ConstantMean()
        self.hyper_prior_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # c) prepare inference
        self._setup_optimizer(optimizer, lr_params, lr_decay)

        self.fitted = False


    def meta_fit(self, valid_tuples=None, verbose=True, log_period=500, n_iter=None):
        """
        meta-learns the GP prior parameters

        Args:
            valid_tuples: list of valid tuples, i.e. [(test_context_x_1, test_context_t_1, test_x_1, test_t_1), ...]
            verbose: (boolean) whether to print training progress
            log_period: (int) number of steps after which to print stats
            n_iter: (int) number of gradient descent iterations
        """
        warnings.simplefilter('ignore', NumericalWarning)

        for task_dict in self.task_dicts: task_dict['model'].train()
        self.likelihood.train()

        assert (valid_tuples is None) or (all([len(valid_tuple) == 4 for valid_tuple in valid_tuples]))

        if len(self.shared_parameters) > 0:
            t = time.time()
            cum_loss = 0.0

            if n_iter is None:
                n_iter = self.num_iter_fit

            for itr in range(1, n_iter + 1):

                loss = 0.0
                self.optimizer.zero_grad()

                for task_dict in self.rds_numpy.choice(self.task_dicts, size=self.task_batch_size):

                    # mll term
                    output = task_dict['model'](task_dict['x_train'])
                    mll = task_dict['mll_fn'](output, task_dict['y_train'])

                    # kl term
                    kl = self._f_kl(task_dict)

                    #  terms for pre-factors
                    n = len(self.task_dicts)
                    m = task_dict['x_train'].shape[0]

                    # loss for this batch
                    loss += - mll / (self.task_batch_size * m) + \
                            self.prior_factor * (1 / math.sqrt(n) + 1 / (n * m)) * kl / self.task_batch_size

                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                cum_loss += loss

                # print training stats stats
                if itr == 1 or itr % log_period == 0:
                    duration = time.time() - t
                    avg_loss = cum_loss / (log_period if itr > 1 else 1.0)
                    cum_loss = 0.0
                    t = time.time()

                    message = 'Iter %d/%d - Loss: %.6f - Time %.2f sec' % (itr, self.num_iter_fit, avg_loss.item(), duration)

                    # if validation data is provided  -> compute the valid log-likelihood
                    if valid_tuples is not None:
                        self.likelihood.eval()
                        valid_ll, valid_rmse, calibr_err, calibr_err_chi2 = self.eval_datasets(valid_tuples)
                        self.likelihood.train()
                        message += ' - Valid-LL: %.3f - Valid-RMSE: %.3f - Calib-Err %.3f' % (valid_ll, valid_rmse, calibr_err)

                    if verbose:
                        self.logger.info(message)

        else:
            self.logger.info('Vanilla mode - nothing to fit')

        self.fitted = True

        for task_dict in self.task_dicts: task_dict['model'].eval()
        self.likelihood.eval()
        return loss.item()

    def predict(self, context_x, context_y, test_x, return_density=False):
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

    def _setup_gp_prior(self, mean_module, covar_module, learning_mode, feature_dim, mean_nn_layers, kernel_nn_layers):

        self.shared_parameters = []

        # a) determine kernel map & module
        if covar_module == 'NN':
            assert learning_mode in ['learn_kernel', 'both'], 'neural network parameters must be learned'
            self.nn_kernel_map = NeuralNetwork(input_dim=self.input_dim, output_dim=feature_dim,
                                          layer_sizes=kernel_nn_layers).to(device)
            self.shared_parameters.append(
                {'params': self.nn_kernel_map.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
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
                {'params': self.nn_mean_fn.parameters(), 'lr': self.lr_params, 'weight_decay': self.weight_decay})
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
            self.shared_parameters.append({'params': self.covar_module.hyperparameters(), 'lr': self.lr_params, 'weight_decay': 0.0})

        if learning_mode in ["learn_mean", "both"] and self.mean_module is not None:
            self.shared_parameters.append({'params': self.mean_module.hyperparameters(), 'lr': self.lr_params, 'weight_decay': 0.0})

    def _setup_optimizer(self, optimizer, lr, lr_decay):
        if optimizer == 'Adam':
            self.optimizer = torch.optim.AdamW(self.shared_parameters, lr=lr, weight_decay=self.weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.shared_parameters, lr=lr)
        else:
            raise NotImplementedError('Optimizer must be Adam or SGD')

        if lr_decay < 1.0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=lr_decay)
        else:
            self.lr_scheduler = DummyLRScheduler()

    def _vectorize_pred_dist(self, pred_dist):
        return torch.distributions.Normal(pred_dist.mean, pred_dist.stddev)

    def _sample_measurement_set(self, x_train):
        if self.train_data_in_kl:
            n_train_x = min(x_train.shape[0], self.num_samples_kl // 2)
            n_rand_x = self.num_samples_kl - n_train_x
            idx_rand = np.random.choice(x_train.shape[0], n_train_x)
            x_kl = torch.cat([x_train[idx_rand], self.domain_dist.sample((n_rand_x,))], dim=0)
        else:
            x_kl = self.domain_dist.sample((self.num_samples_kl,))
        assert x_kl.shape == (self.num_samples_kl, self.input_dim)
        return x_kl

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


if __name__ == "__main__":
    from experiments.data_sim import GPFunctionsDataset, RandomMixture

    data_sim = RandomMixture(random_state=np.random.RandomState(29))
    meta_train_data = data_sim.generate_meta_train_data(n_tasks=20, n_samples=10)
    meta_test_data = data_sim.generate_meta_test_data(n_tasks=50, n_samples_context=10, n_samples_test=160)

    NN_LAYERS = (32, 32, 32, 32)

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

    # for prior_factor in [0.4, 0.2, 0.1, 0.05, 0.02]:
    #     for weight_decay in [0.8, 0.4]:
    for prior_factor in [1.0, 0.4, 0.2, 0.1, 0.05, 0.02]:
        for weight_decay in [0.01]:
            gp_model = FPACOH_MAP(meta_train_data, num_iter_fit=20000, weight_decay=weight_decay, task_batch_size=2,
                                  covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                                  kernel_nn_layers=NN_LAYERS, prior_factor=prior_factor, prior_lengthscale=0.2)
            itrs = 0
            print("---- weight-decay =  %.4f, prior-factor = %.4f ----"%(weight_decay, prior_factor))
            for i in range(4):
                gp_model.meta_fit(valid_tuples=meta_test_data, log_period=500, n_iter=1000)
                itrs += 1000

                x_plot = np.linspace(-10, 10, num=150)
                x_context, t_context, x_test, y_test = meta_test_data[0]
                pred_mean, pred_std = gp_model.predict(x_context, t_context, x_plot)
                ucb, lcb = gp_model.confidence_intervals(x_context, t_context, x_plot, confidence=0.95)

                plt.scatter(x_test, y_test)
                plt.scatter(x_context, t_context)

                plt.plot(x_plot, pred_mean)
                plt.fill_between(x_plot, lcb, ucb, alpha=0.2)
                plt.title('F=PACOH MAP meta mll (wd=  %.4f, prior_weight = %.4f) itrs = %i' % (weight_decay, prior_factor, itrs))
                plt.show()