'''
GPyTorch Implementation of Gaussian Processes
'''
import math
import torch
import gpytorch
from matplotlib import pyplot as plt


torch.manual_seed(0)
class ClassificationGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        super(ClassificationGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        posterior_latent_dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return posterior_latent_dist


class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_tasks, num_inducing_pts):
        inducing_points = torch.rand(num_latents, num_inducing_pts, 1)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )

        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
            batch_shape=torch.Size([num_latents])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        posterior_latent_dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return posterior_latent_dist


class MultitaskBernoulliLikelihood(gpytorch.likelihoods.Likelihood):
    def forward(self, function_samples, **kwargs):
        print(function_samples.shape)
        output_probs = torch.distributions.Normal(0, 1).cdf(function_samples)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=output_probs), 1)


class ClassificationRunner:
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self, inducing_points):
        self.model = ClassificationGPModel(inducing_points)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def train(self, train_x, train_y, training_iter):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                      {'params': self.likelihood.parameters()},], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_x.size(0), beta=1.)
        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(train_x)
            print(f"Output: {output.mean.shape}")
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item()
            ))
            optimizer.step()
    
    def predict(self, test_x):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior_latent_dist = self.model(test_x)
            # probabilities = test_dist.loc # Distribution mean
            posterior_observed_dist = self.likelihood(posterior_latent_dist)
            # Should be shape [num_pts, num_tasks] = [23, 2]
            
        return posterior_latent_dist, posterior_observed_dist


class MultitaskClassificationRunner(ClassificationRunner):
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self, num_latents, num_tasks, num_inducing_points):
        self.model = MultitaskGPModel(num_latents, num_tasks, num_inducing_points)
        #self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.likelihood = MultitaskBernoulliLikelihood()
