import torch


class AbstractScoreEstimator:

    @staticmethod
    def rbf_kernel(x1, x2, bandwidth):
        return torch.exp(-torch.sum(((x1 - x2) / bandwidth)**2, dim=-1) / 2)

    def gram(self, x1, x2, bandwidth):
        """
        x1: [..., n1, D]
        x2: [..., n2, D]
        bandwidth: [..., 1, 1, D]
        returns: [..., n1, n2]
        """
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        return self.rbf_kernel(x_row, x_col, bandwidth)

    def grad_gram(self, x1, x2, bandwidth):
        """
        x1: [..., n1, D]
        x2: [..., n2, D]
        bandwidth: [..., 1, 1, D]
        returns: [..., n1, n2], [..., n1, n2, D], [..., n1, n2, D]
        """
        x_row = torch.unsqueeze(x1, -2)
        x_col = torch.unsqueeze(x2, -3)
        # g: [..., n1, n2]
        g = self.rbf_kernel(x_row, x_col, bandwidth)
        # diff: [..., n1, n2, D]
        diff = (x_row - x_col) / (bandwidth ** 2)
        # g_expand: [..., n1, n2, 1]
        g_expand = torch.unsqueeze(g, -1)
        # grad_x1: [..., n1, n2, D]
        grad_x2 = g_expand * diff
        # grad_x2: [..., n1, n2, D]
        grad_x1 = g_expand * (-diff)
        return g, grad_x1, grad_x2

    @staticmethod
    def median_heuristic(x_samples, x_basis):
        """
        x_samples: [..., n_samples, d]
        x_basis: [..., n_basis, d]
        returns: [..., 1, 1, d]
        """
        d = x_samples.shape[-1]
        n_samples = x_samples.shape[-2]
        n_basis = x_basis.shape[-2]
        x_samples_expand = torch.unsqueeze(x_samples, -2)
        x_basis_expand = torch.unsqueeze(x_basis, -3)
        pairwise_dist = torch.abs(x_samples_expand - x_basis_expand)

        length = len(pairwise_dist.shape)
        # reorder dimensions as [2, 0, 1]
        pairwise_dist = pairwise_dist.transpose(dim0=-3, dim1=-1)
        pairwise_dist = pairwise_dist.transpose(dim0=-2, dim1=-1)

        k = n_samples * n_basis // 2
        k = k if k > 0 else 1
        top_k_values = torch.topk(torch.reshape(pairwise_dist, [-1, d, n_samples * n_basis]), k=k).values
        bandwidth = torch.reshape(top_k_values[:, :, -1], list(x_samples.shape[:-2]) + [1, 1, d])
        bandwidth *= d ** 0.5
        bandwidth += (bandwidth < 1e-6).type(bandwidth.dtype)
        return bandwidth

class SSGE(AbstractScoreEstimator):
    def __init__(self, n_eigen=6, eta=1e-3, n_eigen_threshold=None, bandwidth=2.0):
        self.n_eigen_threshold = n_eigen_threshold
        self.bandwidth = bandwidth
        self.n_eigen = n_eigen
        self.eta = eta

    def nystrom_ext(self, s, x, eigen_vectors, eigen_values, bandwidth):
        """
        s: [..., m, d]
        index_points: [..., n, d]
        eigen_vectors: [..., m, n_eigen]
        eigen_values: [..., n_eigen]
        returns: [..., n, n_eigen], by default n_eigen=m.
        """
        m = s.shape[-2]
        # kxq: [..., N, m]
        kxq = self.gram(x, s, bandwidth)
        # ret: [..., N, n_eigen]
        ret = torch.matmul(kxq, eigen_vectors)
        ret *= torch.tensor(m ** 0.5, dtype=eigen_values.dtype) / torch.unsqueeze(eigen_values, dim=-2)
        return ret

    def estimate_gradients(self, s, x=None):
        if x is not None:
            return torch.transpose(self.__call__(torch.transpose(s, 0, 1), torch.transpose(x, 0, 1)), 0, 1)
        else:
            return torch.transpose(self.__call__(torch.transpose(s, 0, 1)), 0, 1)

    def __call__(self, s, x=None):
        """
        s: [..., m, d], samples
        x: [..., n, d], index points
        """
        if x is None:
            x = stacked_samples = s
        else:
            stacked_samples = torch.cat([s, x], dim=-2)

        if self.bandwidth is None:
            length_scale = self.median_heuristic(stacked_samples, stacked_samples)
        else:
            length_scale = self.bandwidth

        m = s.shape[-2]
        # kq: [..., m, m]
        # grad_k1: [..., m, m, d]
        # grad_k2: [..., m, m, d]
        kq, grad_k1, grad_k2 = self.grad_gram(s, s, length_scale)
        kq += self.eta * torch.eye(m)
        # eigen_vectors: [..., m, m]
        # eigen_values: [..., m]
        eigen_values, eigen_vectors = torch.symeig(kq, eigenvectors=True)
        #eigen_values = eigen_values[:, 0] # only take real part
        if (self.n_eigen is None) and (self.n_eigen_threshold is not None):
            eigen_arr = torch.mean(torch.reshape(eigen_values, [-1, m]), dim=0)
            eigen_arr = torch.flip(eigen_arr, dim=[-1])
            eigen_arr /= torch.sum(eigen_arr)
            eigen_cum = torch.cumsum(eigen_arr, dim=-1)
            eigen_les = (eigen_cum < self.n_eigen_threshold).int()
            self.n_eigen = torch.sum(eigen_les)
        if self.n_eigen is not None:
            eigen_values = eigen_values[..., -self.n_eigen:]
            eigen_vectors = eigen_vectors[..., -self.n_eigen:]
        # eigen_ext: [..., n, n_eigen]
        eigen_ext = self.nystrom_ext(s, x, eigen_vectors, eigen_values, length_scale)
        # grad_k1_avg = [..., m, d]
        grad_k1_avg = torch.mean(grad_k1, dim=-3)
        # beta: [..., n_eigen, d]
        beta = - torch.matmul(torch.transpose(eigen_vectors, 0, 1), grad_k1_avg)
        beta *= torch.tensor(m ** 0.5, dtype=eigen_values.dtype)  / torch.unsqueeze(eigen_values, dim=-1)
        # grads: [..., n, d]
        grads = torch.matmul(eigen_ext, beta)
        return grads

if __name__ == '__main__':
    import numpy as np

    def frob_dist(A, B):
        return np.sum((A.detach().numpy() - B)**2)

    ssge = SSGE()

    x_sample = np.expand_dims(np.arange(-2, 2, step=0.1), axis=-1)
    x_sample_torch = torch.from_numpy(x_sample)

    b_torch = ssge.median_heuristic(x_sample_torch, x_sample_torch)


    K_torch, g1_torch, g2_torch = ssge.grad_gram(x_sample_torch, x_sample_torch, bandwidth=1.0)
    score_torch = ssge(x_sample_torch)

    x_plot = np.random.normal(size=(100, 4))
    x_plot_torch = torch.from_numpy(x_plot)
