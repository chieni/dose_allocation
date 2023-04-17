import math
import torch
import gpytorch
import tqdm
from matplotlib import pyplot as plt


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_classes):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size((num_classes,)))
        self.covar_module = gpytorch.kernels.RBFKernel(batch_shape=torch.Size((num_classes,)))

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1, batch_shape=torch.Size((num_classes,)))

    def forward(self,x,i):
        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)



train_x1 = torch.rand(2000)
train_x2 = torch.rand(1000)

train_y1 = ((torch.sin(train_x1 * (2 * math.pi)) + torch.randn(train_x1.size()) * 0.2) > 0) + 0.
train_y2 = ((torch.cos(train_x2 * (2 * math.pi)) + torch.randn(train_x2.size()) * 0.2) > 0) + 0.

train_i_task1 = torch.full((train_x1.shape[0],1), dtype=torch.long, fill_value=0)
train_i_task2 = torch.full((train_x2.shape[0],1), dtype=torch.long, fill_value=1)

full_train_x = torch.cat([train_x1, train_x2])
full_train_i = torch.cat([train_i_task1, train_i_task2])
full_train_y = torch.cat([train_y1, train_y2]).long()

# Transform likelihood targets
likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(full_train_y, learn_additional_noise=True)

# Here we have two iterms that we're passing in as train_inputs
model = MultitaskGPModel((full_train_x, full_train_i), likelihood.transformed_targets, likelihood, num_classes=2)

training_iterations = 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(full_train_x, full_train_i)
    loss = -mll(output, likelihood.transformed_targets).sum()
    loss.backward()
    print('Iter %d/50 - Loss: %.3f' % (i + 1, loss.item()))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

# Test points every 0.02 in [0,1]
test_x = torch.linspace(0, 1, 51)
test_i_task1 = torch.LongTensor([0 for item in range(test_x.shape[0])])
test_i_task2 = torch.LongTensor([1 for item in range(test_x.shape[0])])

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_dist_y1 = model(test_x, test_i_task1)
    test_dist_y2 = model(test_x, test_i_task2)

    observed_pred_y1 = likelihood(model(test_x, test_i_task1))
    observed_pred_y2 = likelihood(model(test_x, test_i_task2))

pred_samples_y1 = test_dist_y1.sample(torch.Size((256,))).exp()
probabilities_y1 = (pred_samples_y1 / pred_samples_y1.sum(-2, keepdim=True)).mean(0)

pred_samples_y2 = test_dist_y2.sample(torch.Size((256,))).exp()
probabilities_y2 = (pred_samples_y2 / pred_samples_y2.sum(-2, keepdim=True)).mean(0)

# Define plotting function
def ax_plot(ax, train_y, train_x, posterior_probs, test_x, title):
    # Get lower and upper confidence bounds
    # lower, upper = posterior_dist.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.detach().numpy(), train_y.detach().numpy(), 'k*')
    # Predictive mean as blue line
    ax.plot(test_x.numpy(), posterior_probs, 'b')
    # Shade in confidence
    # ax.fill_between(test_x.detach().numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    ax.set_title(title)


# Plot both tasks
# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))
ax_plot(y1_ax, train_y1, train_x1, probabilities_y1[1], test_x, 'Observed Values (Likelihood)')
ax_plot(y2_ax, train_y2, train_x2, probabilities_y2[1], test_x, 'Observed Values (Likelihood)')
plt.show()