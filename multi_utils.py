import os
import torch
import numpy as np
np.random.seed(0)
import random
random.seed(0)
torch.manual_seed(0)
from botorch.test_functions.multi_objective import C2DTLZ2
import tqdm
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import sys
from time import sleep
import pandas as pd

class ExcelHandler:
     def __init__(self, filename):
        self.filename = filename
        self.variable_info = self._read_variable_info()
        self.feature_vars = self.variable_info[self.variable_info['Settings'] == 'Feature']
        self.constraint_vars = self.variable_info[self.variable_info['Settings'] == 'Constraint']
        self.target_vars = self.variable_info[self.variable_info['Settings'] == 'Target']  # New attribute
        self.nfeat = len(self.feature_vars)
        self.ncons = len(self.constraint_vars)
        self.ntargets = len(self.target_vars)  # New attribute

     def _read_variable_info(self):
        xl = pd.ExcelFile(self.filename)
        description_df = xl.parse('Description')
        return description_df

     def get_bounds(self):
        bounds_np = self.feature_vars[['Lower Bound', 'Upper Bound']].values
        bounds_tensor = torch.tensor(bounds_np, dtype=torch.float32)
        return bounds_tensor.t()

     def observations_to_tensors(self):
        # Read the 'Observations' sheet
        xl = pd.ExcelFile(self.filename)
        observations_df = xl.parse('Observations', header=0)
        # Extract input variables and responses
        features_df = observations_df[self.feature_vars['Variable Name'].values]
        responses_df = observations_df[self.target_vars['Variable Name'].values]  # Updated to use target_vars
        # Convert input variables and responses to PyTorch tensors
        train_x = torch.tensor(features_df.values, dtype=torch.float32)
        train_y = torch.tensor(responses_df.values, dtype=torch.float32)  # Removed squeeze() because we have multiple targets
        return train_x, train_y

     def load_excel(self):
        self.variable_info = self._read_variable_info()
        self.feature_vars = self.variable_info[self.variable_info['Settings'] == 'Feature']
        self.constraint_vars = self.variable_info[self.variable_info['Settings'] == 'Constraint']
        self.nfeat = len(self.feature_vars)
        self.ncons = len(self.constraint_vars)



class MultiObjectiveBO:
    def __init__(self, train_x, train_y, bounds, batch_size=1, reference_point=None):
        self.train_x = train_x.to(torch.float64)
        self.train_y = train_y.to(torch.float64)
        self.bounds = bounds.to(torch.float64)
        self.batch_size = batch_size

        self.tkwargs = {
            "dtype": torch.double,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }
        self.NUM_RESTARTS = 10
        self.RAW_SAMPLES = 512

        #self.standard_bounds = torch.zeros(2, self.bounds.size(0), **self.tkwargs)
        #self.standard_bounds[1] = 1

        # If a reference point is not provided, use the minimum of the train_y as reference point
        self.ref_point = reference_point if reference_point is not None else torch.min(train_y, dim=0).values

        # Initialize the model
        self.model = self.train_GPs(self.train_x, self.train_y)
        # Fit the model
        fit_gpytorch_model(self.model[0])

    def train_GPs(self, train_x, train_y):
        model = SingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=train_y.shape[-1]))
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def get_posterior_stats(self, test_x):
        # Set the model to evaluation mode
        self.model[1].eval()
        # Disable gradient calculations
        torch.set_grad_enabled(False)
        # Predict the posterior mean and variance for each objective
        with torch.no_grad():
            posterior = self.model[1].posterior(test_x)
            posterior_mean = posterior.mean
            posterior_variance = posterior.variance
        # Restore gradient calculations
        torch.set_grad_enabled(True)
        return posterior_mean, torch.sqrt(posterior_variance)

    def get_pareto_points(self,tensor_y):
        is_efficient_mask = is_non_dominated(tensor_y)
        pareto_points = tensor_y[is_efficient_mask]
        return pareto_points

    def compute_hypervolume(self, tensor_y: torch.Tensor) -> float:
        pareto_points = self.get_pareto_points(tensor_y)
        hv = Hypervolume(ref_point=self.ref_point)
        return hv.compute(pareto_points)

    def optimize_acquisition(self):
        # Create a QMC sampler
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        partitioning = NondominatedPartitioning(Y=self.train_y, ref_point=self.ref_point)
        acq_func = qExpectedHypervolumeImprovement(
            model=self.model[1],
            ref_point=self.ref_point.tolist(),  # use known reference point
            partitioning=partitioning,
            sampler=sampler,
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,#self.standard_bounds,
            q=self.batch_size,
            num_restarts=self.NUM_RESTARTS,
            raw_samples=self.RAW_SAMPLES,  # used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
            sequential=True,
        )

        # observe new values
        #new_x = unnormalize(candidates.detach(), bounds=self.bounds)
        new_x = candidates.detach()
        return new_x
