import os
import numpy as np
import torch
import lib.utils as utils


class Lorenz96(object):

	T          = 200
	D          = 30
	F_FORCING  = 8.0
	T_TOTAL    = 5.0
	NOISE = 0.2

	n_training_samples = 10000
	training_file = 'training.pt'

	def __init__(self, root, generate=False, noise_weight=NOISE, device=torch.device("cpu")):
		self.root = root

		if generate:
			self._generate_dataset()

		if not self._check_exists():
			raise RuntimeError('Dataset not found. Use generate=True to generate it.')

		data_file = os.path.join(self.data_folder, self.training_file)
		self.data = torch.Tensor(torch.load(data_file, weights_only=False)).to(device)
		self.data, self.data_min, self.data_max = utils.normalize_data(self.data)
		if noise_weight > 0:
			self.data += noise_weight * torch.randn_like(self.data)
		self.device = device

	def _generate_dataset(self):
		if self._check_exists():
			return
		os.makedirs(self.data_folder, exist_ok=True)
		print('Generating Lorenz-96 dataset...')
		data = self._generate_random_trajectories(self.n_training_samples)
		torch.save(data, os.path.join(self.data_folder, self.training_file))
		print('Done.')

	def _generate_random_trajectories(self, n_samples):
		try:
			from scipy.integrate import solve_ivp
		except ImportError as e:
			raise Exception('scipy is required to generate the dataset.') from e

		np.random.seed(42)
		t_eval = np.linspace(0, self.T_TOTAL, self.T)

		x0 = self.F_FORCING * np.ones((n_samples, self.D))
		x0 += np.random.randn(n_samples, self.D) * 2.0

		def lorenz96_batch(t, x_flat):
			x    = x_flat.reshape(n_samples, self.D)
			xp1  = np.roll(x, -1, axis=1)
			xm1  = np.roll(x,  1, axis=1)
			xm2  = np.roll(x,  2, axis=1)
			dxdt = (xp1 - xm2) * xm1 - x + self.F_FORCING
			return dxdt.flatten()

		print(f'Integrating {n_samples} Lorenz-96 trajectories...')
		sol = solve_ivp(
			lorenz96_batch,
			t_span=(0, self.T_TOTAL),
			y0=x0.flatten(),
			t_eval=t_eval,
			method='RK45',
			rtol=1e-6,
			atol=1e-6,
		)

		# sol.y: (n_samples * D, T)  →  (n_samples, T, D)
		data = sol.y.T.reshape(self.T, n_samples, self.D)
		data = data.transpose(1, 0, 2).astype(np.float32)
		return data

	def _check_exists(self):
		return os.path.exists(os.path.join(self.data_folder, self.training_file))

	@property
	def data_folder(self):
		return os.path.join(self.root, self.__class__.__name__)

	def get_dataset(self):
		return self.data

	def __len__(self):
		return len(self.data)

	def size(self, ind=None):
		if ind is not None:
			return self.data.shape[ind]
		return self.data.shape

	def __repr__(self):
		fmt_str  = 'Dataset ' + self.__class__.__name__ + '\n'
		fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
		fmt_str += '    Root Location: {}\n'.format(self.root)
		return fmt_str
