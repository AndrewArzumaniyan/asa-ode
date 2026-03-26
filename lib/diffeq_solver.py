###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import time
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from torch.distributions.multivariate_normal import MultivariateNormal

# git clone https://github.com/rtqichen/torchdiffeq.git
try:
	from torchdiffeq import odeint as odeint
except ModuleNotFoundError:
	def odeint(func, y0, time_steps, rtol=None, atol=None, method=None):
		# Minimal fallback integrator for local runs when torchdiffeq is unavailable.
		if len(time_steps) == 1:
			return y0.unsqueeze(0)

		ys = [y0]
		y = y0

		for i in range(1, len(time_steps)):
			t0 = time_steps[i - 1]
			t1 = time_steps[i]
			dt = t1 - t0

			if method == "euler":
				y = y + dt * func(t0, y)
			else:
				k1 = func(t0, y)
				k2 = func(t0 + dt / 2, y + dt * k1 / 2)
				k3 = func(t0 + dt / 2, y + dt * k2 / 2)
				k4 = func(t1, y + dt * k3)
				y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

			ys.append(y)

		return torch.stack(ys, dim=0)

#####################################################################################################

class DiffeqSolver(nn.Module):
	def __init__(self, input_dim, ode_func, method, latents, 
			odeint_rtol = 1e-4, odeint_atol = 1e-5, device = torch.device("cpu")):
		super(DiffeqSolver, self).__init__()

		self.ode_method = method
		self.latents = latents		
		self.device = device
		self.ode_func = ode_func

		self.odeint_rtol = odeint_rtol
		self.odeint_atol = odeint_atol

	def forward(self, first_point, time_steps_to_predict, backwards = False):
		"""
		# Decode the trajectory through ODE Solver
		"""
		n_traj_samples, n_traj = first_point.size()[0], first_point.size()[1]
		n_dims = first_point.size()[-1]

		pred_y = odeint(self.ode_func, first_point, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		pred_y = pred_y.permute(1,2,0,3)

		assert(torch.mean(pred_y[:, :, 0, :]  - first_point) < 0.001)
		assert(pred_y.size()[0] == n_traj_samples)
		assert(pred_y.size()[1] == n_traj)

		return pred_y

	def sample_traj_from_prior(self, starting_point_enc, time_steps_to_predict, 
		n_traj_samples = 1):
		"""
		# Decode the trajectory through ODE Solver using samples from the prior

		time_steps_to_predict: time steps at which we want to sample the new trajectory
		"""
		func = self.ode_func.sample_next_point_from_prior

		pred_y = odeint(func, starting_point_enc, time_steps_to_predict, 
			rtol=self.odeint_rtol, atol=self.odeint_atol, method = self.ode_method)
		# shape: [n_traj_samples, n_traj, n_tp, n_dim]
		pred_y = pred_y.permute(1,2,0,3)
		return pred_y


