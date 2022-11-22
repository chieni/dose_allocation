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
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ClassificationGPModel, self).__init__(variational_strategy)
        # TODO: change to linear mean?
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


class MultitaskSubgroupGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(MultitaskSubgroupGPModel, self).__init__(variational_strategy)
        # TODO: change to linear mean?
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1)

        # inducing_points = torch.rand(num_latents, num_inducing_pts, 1)
        # variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
        #     inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        # )
        # variational_strategy = gpytorch.variational.LMCVariationalStrategy(
        #     gpytorch.variational.VariationalStrategy(
        #         self, inducing_points, variational_distribution, learn_inducing_locations=True
        #     ),
        #     num_tasks=num_tasks,
        #     num_latents=num_latents,
        #     latent_dim=-1
        # )
        # super().__init__(variational_strategy)
        # self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])),
        #     batch_shape=torch.Size([num_latents])
        # )
        # self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=num_tasks, rank=1)

    def forward(self, x, task_indices):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        import pdb; pdb.set_trace()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_i = self.task_covar_module(task_indices)
        covar = covar_x.mul(covar_i)
        posterior_latent_dist = gpytorch.distributions.MultivariateNormal(mean_x, covar)
        return posterior_latent_dist
    

class MultitaskBernoulliLikelihood(gpytorch.likelihoods.Likelihood):
    def forward(self, function_samples, **kwargs):
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

    def train(self, train_x, train_y, num_epochs):
        self.model.train()
        self.likelihood.train()
        # for param_name, param in self.model.named_parameters():
        #     print(f"Parameter name: {param_name} value: {param}")
        optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                      {'params': self.likelihood.parameters()},], lr=0.01)
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


class MultitaskClassificationRunner:
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self, num_latents, num_tasks, num_inducing_points):
        self.model = MultitaskGPModel(num_latents, num_tasks, num_inducing_points)
        self.likelihood = MultitaskBernoulliLikelihood()

    def train(self, train_x, train_y, num_epochs):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                      {'params': self.likelihood.parameters()},], lr=0.1)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0), beta=1.)
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
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

class MultitaskSubgroupClassificationRunner:
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self, inducing_points, num_tasks):
        self.model = MultitaskSubgroupGPModel(inducing_points, num_tasks)
        self.likelihood = MultitaskBernoulliLikelihood()

    def train(self, train_x, train_y, task_indices, num_epochs):
        self.model.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam([{'params': self.model.parameters()},
                                      {'params': self.likelihood.parameters()},], lr=0.1)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0), beta=1.)
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = self.model(train_x, task_indices)
            loss = -mll(output, train_y)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
    
    def predict(self, test_x, task_indices):
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior_latent_dist = self.model(test_x, task_indices)
            posterior_observed_dist = self.likelihood(posterior_latent_dist)
            
        return posterior_latent_dist, posterior_observed_dist