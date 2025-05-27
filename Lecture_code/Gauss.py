import torch
from torch.distributions import constraints
from torch.nn import Parameter
import numpy as np
import pyro 
import pyro.contrib.gp as gp
import pyro.distributions as dist
from pyro.contrib.gp.models.model import GPModel
from pyro.contrib.gp.util import conditional
from pyro.distributions.util import eye_like
from pyro.nn.module import PyroParam, pyro_method
from matplotlib import pyplot as plt
import pandas as pd

# df = pd.read_csv(r'csv_files\df_baseline.csv')
# feature_cols = [
#     'Levy', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags',
#     'Turbo', 'Age',
#     'Manufacturer_encoded', 'Category_encoded', 'GearBox_encoded',
#     'Drive_4x4', 'Drive_Front', 'Drive_Rear'
# ]
# target_col = 'Price'

# # Drop rows with missing values if any (optional)
# df = df.dropna(subset=feature_cols + [target_col])

# # Extract features and target
# X = df[feature_cols].values
# y = df[target_col].values

# Convert to torch tensors
# X_train_tensor = torch.tensor(X, dtype=torch.float32)
# y_train_tensor = torch.tensor(y, dtype=torch.float32)
X_train = np.hstack([-0.75 + np.random.rand(1,20), 0.75 + np.random.rand(1,20)]).T
y_train = np.sin(4.0*X_train) + 0.1*np.random.randn(len(X_train), 1)
N_train = len(y_train)
# Build torch.tensors
X_train_tensor = torch.from_numpy(X_train).view(-1,)
y_train_tensor = torch.from_numpy(y_train).view(-1,)


class VariationalGP(GPModel):
    def __init__(self, X, y, kernel, likelihood, mean_function=None,
                 latent_shape=None, whiten=False, jitter=1e-6):
        super().__init__(X, y, kernel, mean_function, jitter)

        self.likelihood = likelihood

        y_batch_shape = self.y.shape[:-1] if self.y is not None else torch.Size([])
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape

        N = self.X.size(0)
        f_loc = self.X.new_zeros(self.latent_shape + (N,))
        self.f_loc = Parameter(f_loc)

        identity = eye_like(self.X, N)
        f_scale_tril = identity.repeat(self.latent_shape + (1, 1))
        self.f_scale_tril = PyroParam(f_scale_tril, constraints.lower_cholesky)

        self.whiten = whiten
        self._sample_latent = True

    @pyro_method
    def model(self):
        self.set_mode("model")

        N = self.X.size(0)
        Kff = self.kernel(self.X).contiguous()
        Kff.view(-1)[::N + 1] += self.jitter  # add jitter to the diagonal
        Lff = torch.linalg.cholesky(Kff)

        zero_loc = self.X.new_zeros(self.f_loc.shape)
        if self.whiten:
            identity = eye_like(self.X, N)
            pyro.sample(self._pyro_get_fullname("f"),
                        dist.MultivariateNormal(zero_loc, scale_tril=identity)
                            .to_event(zero_loc.dim() - 1))
            f_scale_tril = Lff.matmul(self.f_scale_tril)
            f_loc = Lff.matmul(self.f_loc.unsqueeze(-1)).squeeze(-1)
        else:
            pyro.sample(self._pyro_get_fullname("f"),
                        dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                            .to_event(zero_loc.dim() - 1))
            f_scale_tril = self.f_scale_tril
            f_loc = self.f_loc

        f_loc = f_loc + self.mean_function(self.X)
        f_var = f_scale_tril.pow(2).sum(dim=-1)
        if self.y is None:
            return f_loc, f_var
        else:
            return self.likelihood(f_loc, f_var, self.y)

    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self._load_pyro_samples()

        pyro.sample(self._pyro_get_fullname("f"),
                    dist.MultivariateNormal(self.f_loc, scale_tril=self.f_scale_tril)
                        .to_event(self.f_loc.dim()-1))
        

    def forward(self, Xnew, full_cov=False):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:
        .. math:: p(f^* \mid X_{new}, X, y, k, f_{loc}, f_{scale\_tril})
            = \mathcal{N}(loc, cov).
        .. note:: Variational parameters ``f_loc``, ``f_scale_tril``, together with
            kernel's parameters have been learned from a training procedure (MCMC or
            SVI).
        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)
        self.set_mode("guide")

        loc, cov = conditional(Xnew, self.X, self.kernel, self.f_loc, self.f_scale_tril,
                               full_cov=full_cov, whiten=self.whiten, jitter=self.jitter)
        return loc + self.mean_function(Xnew), cov
    
def plot(plot_observed_data=False, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):

        plt.figure(figsize=(12, 6))
        if plot_observed_data:
            plt.plot(X_train_tensor.view(-1,).numpy(), y_train_tensor.view(-1,).numpy(), 'kx')
        if plot_predictions:
            Xtest = torch.linspace(-4, 4, n_test)  # test inputs
            # compute predictive mean and variance
            with torch.no_grad():
                if type(model) == gp.models.VariationalSparseGP:
                    mean, cov = model(Xtest, full_cov=True)
                else:
                    mean, cov = model(Xtest.double(), full_cov=True)
            sd = cov.diag().sqrt()  # standard deviation at each input point x
            plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
            plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                            (mean - 2.0 * sd).numpy(),
                            (mean + 2.0 * sd).numpy(),
                            color='C0', alpha=0.3)
            plt.legend(["Data", "GP Posterior Mean"])
        if n_prior_samples > 0:  # plot samples from the GP prior
            Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
            noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                    else model.likelihood.variance)
            cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
            samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                        .sample(sample_shape=(n_prior_samples,))
            plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

        plt.xlim(-4, 4)
        plt.show()

# initialize the kernel, likelihood, and model
pyro.clear_param_store()
kernel = gp.kernels.RBF(input_dim=1)
likelihood = gp.likelihoods.Gaussian()
# turn on "whiten" flag for more stable optimization
vsgp = VariationalGP(X_train_tensor.view(-1,), y_train_tensor.view(-1,), kernel, likelihood=likelihood, whiten=True)

# instead of defining our own training loop, we will
# use the built-in support provided by the GP module
num_steps = 1500
losses = gp.util.train(vsgp, num_steps=num_steps)
plt.plot(losses)
plt.show()
plot(model=vsgp, plot_observed_data=True, plot_predictions=True)