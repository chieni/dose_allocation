'''
GPyTorch Implementation of Gaussian Processes
'''
import math
import torch
import gpytorch
from matplotlib import pyplot as plt


torch.manual_seed(0)
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train(train_x, train_y):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    training_iter = 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    return model, likelihood

def predict(model, likelihood, train_x, train_y):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 51)
        observed_pred = likelihood(model(test_x))

    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()

def regression_example():
    # Basic regression
    train_x = torch.linspace(0, 1, 100)
    train_y = torch.sin(train_x * (2*math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
    model, likelihood = train(train_x, train_y)
    predict(model, likelihood, train_x, train_y)


class MultitaskBernoulliLikelihood(gpytorch.likelihoods.Likelihood):
    def forward(self, function_samples, **kwargs):
        output_probs = torch.distributions.Normal(0, 1).cdf(function_samples)
        return torch.distributions.Independent(torch.distributions.Bernoulli(probs=output_probs), 1)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

class MultitaskApproxGPModel(gpytorch.models.ApproximateGP):
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


train_x1 = torch.rand(2000)
train_x2 = torch.rand(1000)

train_y1 = ((torch.sin(train_x1 * (2 * math.pi)) + torch.randn(train_x1.size()) * 0.2) > 0) + 0.
train_y2 = ((torch.cos(train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2) > 0) + 0.

# likelihood = gpytorch.likelihoods.GaussianLikelihood()
#likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
likelihood = MultitaskBernoulliLikelihood()

# train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
# train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)

# train_i_task1 = torch.ones(train_x1.shape[0], dtype=torch.int64) * 0.
# train_i_task2 = torch.ones(train_x2.shape[0], dtype=torch.int64)

train_i_task1 = torch.LongTensor([0 for item in range(train_x1.shape[0])])
train_i_task2 = torch.LongTensor([1 for item in range(train_x2.shape[0])])

full_train_x = torch.cat([train_x1, train_x2])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y1, train_y2])

# Here we have two iterms that we're passing in as train_inputs
#model = MultitaskGPModel((full_train_x, full_train_i), full_train_y, likelihood)
model = MultitaskApproxGPModel(3, 2, 5)

training_iterations = 3
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=full_train_y.size(0))


for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(full_train_x, task_indices=full_train_i)
    loss = -mll(output, full_train_y)
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

# Test points every 0.02 in [0,1]
test_x = torch.linspace(0, 1, 51)
test_i_task1 = torch.LongTensor([0 for item in range(test_x.shape[0])])
test_i_task2 = torch.LongTensor([1 for item in range(test_x.shape[0])])

# Make predictions - one task at a time
# We control the task we cae about using the indices

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    posterior_latent_dist1 = model(test_x, task_indices=test_i_task1)
    posterior_latent_dist2 = model(test_x, task_indices=test_i_task2)

    observed_pred_y1 = likelihood(posterior_latent_dist1)
    observed_pred_y2 = likelihood(posterior_latent_dist2)
    

def get_bernoulli_confidence_region(test_x, posterior_latent_dist, likelihood_model, num_samples):
    samples = posterior_latent_dist.sample_n(num_samples)
    likelihood_samples = likelihood_model(samples)
    lower = torch.quantile(likelihood_samples.mean, 0.025, axis=0)
    upper = torch.quantile(likelihood_samples.mean, 1 - 0.025, axis=0)
    mean = likelihood_samples.mean.mean(axis=0)
    variance = likelihood_samples.mean.var(axis=0)
    return mean, lower, upper


# Define plotting function
def ax_plot(ax, train_y, train_x, latent_dist, test_x, likelihood_model, title):
    # Get lower and upper confidence bounds
    # lower, upper = rand_var.confidence_region()
    mean, lower, upper = get_bernoulli_confidence_region(test_x, latent_dist, likelihood_model, 10000)
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.detach().numpy(), mean.detach().numpy(), 'b')
    # Shade in confidence
    ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)

# Plot both tasks
ax_plot(y1_ax, train_y1, train_x1, posterior_latent_dist1, test_x, likelihood, 'Observed Values (Likelihood)')
ax_plot(y2_ax, train_y2, train_x2, posterior_latent_dist2, test_x, likelihood, 'Observed Values (Likelihood)')
plt.show()