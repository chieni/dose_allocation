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
    def __init__(self, num_latents, num_tasks, inducing_points):
        #inducing_points = torch.rand(num_latents, num_inducing_pts, 1)
        inducing_points = inducing_points[[1, 3]]
        inducing_points = torch.tensor(inducing_points)
        inducing_points = inducing_points.repeat(num_latents, 1)
        inducing_points = torch.unsqueeze(inducing_points, 2)

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
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents])) + gpytorch.kernels.LinearKernel(),
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
    def __init__(self, num_latents, num_tasks, inducing_points):
        self.model = MultitaskGPModel(num_latents, num_tasks, inducing_points)
        self.likelihood = MultitaskBernoulliLikelihood()

    def train(self, train_x, train_y, task_indices, num_epochs, learning_rate=0.1, use_gpu=True):
        if use_gpu:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            task_indices = task_indices.cuda()

        self.model.train()
        self.likelihood.train()

        init_lengthscale = 1
        init_variance = 1
        self.model.covar_module.base_kernel.kernels[0].lengthscale = init_lengthscale
        self.model.covar_module.base_kernel.kernels[1].variance = init_variance
        all_params = set(self.model.parameters())
        model_params = list(all_params - {self.model.covar_module.base_kernel.kernels[0].raw_lengthscale, 
                                          self.model.covar_module.base_kernel.kernels[1].raw_variance})
        model_params = self.model.parameters()

        # for name, param in self.model.named_parameters():
        #     print(name, param.data)
        
        optimizer = torch.optim.Adam([{'params': model_params},
                                      {'params': self.likelihood.parameters()},], lr=learning_rate)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0), beta=1.)
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            optimizer.zero_grad()
            output = self.model(train_x, task_indices=task_indices)
            loss = -mll(output, train_y)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
    
    def predict(self, test_x, task_indices):
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior_latent_dist = self.model(test_x, task_indices=task_indices)
            posterior_observed_dist = self.likelihood(posterior_latent_dist)
            
        return posterior_latent_dist, posterior_observed_dist