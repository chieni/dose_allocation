'''
GPyTorch Implementation of Gaussian Processes
'''
import math
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt


torch.manual_seed(0)
class ClassificationGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        inducing_points = torch.tensor(inducing_points)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        
        super(ClassificationGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        posterior_latent_dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return posterior_latent_dist

class ClassificationRunner:
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self, inducing_points, mean_init, lengthscale_init):
        self.model = ClassificationGPModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        self.mean_init = mean_init
        self.lengthscale_init = lengthscale_init

    def train(self, train_x, train_y, num_epochs, learning_rate):
        self.model.train()
        self.likelihood.train()
        # for param_name, param in self.model.named_parameters():
        #     print(f"Parameter name: {param_name} value: {param}")

        model_params = self.model.parameters()
        self.model.mean_module.constant = self.mean_init
        self.model.covar_module.base_kernel.lengthscale = self.lengthscale_init
        self.model.covar_module.base_kernel.outputscale = 1.
        model_params = list(set(model_params) - {self.model.covar_module.base_kernel.raw_lengthscale})
        model_params = list(set(model_params) - {self.model.covar_module.raw_outputscale})
        model_params = list(set(model_params) - {self.model.mean_module.raw_constant})


        optimizer = torch.optim.Adam([{'params': model_params},
                                      {'params': self.likelihood.parameters()},], lr=learning_rate)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_x.size(0), beta=1.)
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()

    def predict(self, test_x):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior_latent_dist = self.model(test_x)
            posterior_observed_dist = self.likelihood(posterior_latent_dist)

        return posterior_latent_dist, posterior_observed_dist