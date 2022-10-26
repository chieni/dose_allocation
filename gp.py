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
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ClassificationGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ClassificationRunner:
    '''
    Runner for ClassificationGPModel
    Train and predict on a Gaussian process model for binary classification
    '''
    def __init__(self):
        pass

    def train(self, train_x, train_y, training_iter):
        model = ClassificationGPModel(train_x)
        likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()},], lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0), beta=1.)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (
                i + 1, training_iter, loss.item()
            ))
            optimizer.step()
        return model, likelihood
    
    def predict(self, model, likelihood, test_x):
        model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_dist = model(test_x)
            probabilities = test_dist.loc # Distribution mean
        return probabilities

