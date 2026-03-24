import os
import numpy as np
import torch
import lib.utils as utils


class CoupledOscillators(object):

	T      = 200
	N      = 15      # осцилляторов → D = 2*N каналов (позиции + скорости)
	OMEGA  = 1.0     # собственная частота
	K      = 0.5     # жёсткость связи
	GAMMA  = 0.1     # затухание
	T_TOTAL = 20.0
	NOISE = 0.2

	n_training_samples = 10000
	training_file = 'training.pt'

	@property
	def D(self):
		return 2 * self.N

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
		print('Generating CoupledOscillators dataset...')
		data = self._generate_random_trajectories(self.n_training_samples)
		torch.save(data, os.path.join(self.data_folder, self.training_file))
		print('Done.')

	def _generate_random_trajectories(self, n_samples):
		try:
			from scipy.integrate import solve_ivp
		except ImportError as e:
			raise Exception('scipy is required to generate the dataset.') from e

		np.random.seed(123)
		t_eval = np.linspace(0, self.T_TOTAL, self.T)

		N, OMEGA, K, GAMMA = self.N, self.OMEGA, self.K, self.GAMMA

		def ode(_, y):
			x = y[:N]
			v = y[N:]
			coupling = np.roll(x, -1) - 2 * x + np.roll(x, 1)
			dx = v
			dv = -OMEGA**2 * x + K * coupling - GAMMA * v
			return np.concatenate([dx, dv])

		data = np.zeros((n_samples, self.T, self.D), dtype=np.float32)
		print(f'Integrating {n_samples} coupled oscillator trajectories...')
		for i in range(n_samples):
			if i % 1000 == 0:
				print(f'  {i}/{n_samples}')
			x0 = np.random.randn(N) * 0.5
			v0 = np.random.randn(N) * 0.5
			y0 = np.concatenate([x0, v0])
			sol = solve_ivp(ode, (0, self.T_TOTAL), y0, t_eval=t_eval, method='RK45')
			data[i] = sol.y.T.astype(np.float32)

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
