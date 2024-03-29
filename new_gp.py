'''
GPyTorch Implementation of Gaussian Processes
'''
import math
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt


torch.manual_seed(0)

class MultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, num_tasks, inducing_points):
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
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]),
        #                                lengthscale_prior=lengthscale_prior) + \
        #     gpytorch.kernels.LinearKernel(batch_shape=torch.Size([num_latents])),
        #     batch_shape=torch.Size([num_latents]),
        #     outputscale_prior=outputscale_prior
        # )
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
        output_probs = torch.distributions.Normal(0, 1).cdf(function_samples)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=output_probs), 1)



class MultitaskClassificationRunner:
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self, num_latents, num_tasks, inducing_points, lengthscale_init, mean_init):
        self.model = MultitaskGPModel(num_latents, num_tasks, inducing_points)
        self.likelihood = MultitaskBernoulliLikelihood()
        self.lengthscale_init = lengthscale_init
        self.mean_init = mean_init
        self.num_latents = num_latents

    def train(self, train_x, train_y, task_indices, num_epochs, learning_rate, use_gpu, set_lmc):
        if self.num_latents == 2:
            # lmc_coeffs = torch.tensor([[0.7, 0.3], [0.3, 0.7]])
            lmc_coeffs = torch.tensor([[1.0, 0.], [0., 1.0]])
        else:
            lmc_coeffs = torch.tensor([[1.0, 0.], [0.2, 0.2], [0., 1.]])
            # lmc_coeffs = torch.tensor([[0.7, 0.3], [0.3, 0.7], [0.3, 0.7]])
        
        if use_gpu:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            task_indices = task_indices.cuda()
            lmc_coeffs = lmc_coeffs.cuda()

        self.model.train()
        self.likelihood.train()

        model_params = self.model.parameters()
        self.model.mean_module.constant = self.mean_init
        self.model.covar_module.base_kernel.lengthscale = self.lengthscale_init
        self.model.covar_module.base_kernel.outputscale = 1.
        self.model.variational_strategy.lmc_coefficients = torch.nn.Parameter(lmc_coeffs)
        
        model_params = list(set(model_params) - {self.model.covar_module.base_kernel.raw_lengthscale})
        model_params = list(set(model_params) - {self.model.covar_module.raw_outputscale})
        model_params = list(set(model_params) - {self.model.mean_module.raw_constant})
        if set_lmc:
            model_params = list(set(model_params) - {self.model.variational_strategy.lmc_coefficients})
        
        optimizer = torch.optim.Adam([{'params': model_params},
                                      {'params': self.likelihood.parameters()},], lr=learning_rate)

        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))
        epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for i in epochs_iter:
            optimizer.zero_grad()
            output = self.model(train_x, task_indices=task_indices)
            loss = -mll(output, train_y)
            epochs_iter.set_postfix(loss=loss.item())
            loss.backward()
            optimizer.step()
    
        # for name, param in self.model.named_parameters():
        #     print(name, param.data)
    
    def predict(self, test_x, task_indices, use_gpu):
        self.model.eval()
        self.likelihood.eval()

        if use_gpu:
            test_x = test_x.cuda()
            task_indices = task_indices.cuda()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior_latent_dist = self.model(test_x, task_indices=task_indices)
            posterior_observed_dist = self.likelihood(posterior_latent_dist)
            
        return posterior_latent_dist, posterior_observed_dist