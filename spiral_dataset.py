import os
import numpy as np
import torch
import lib.utils as utils
from spiral_matrix import generate_spiral2d_matrix


class SpiralDataset(object):

	T            = 200   # train_steps + future_steps
	TRAIN_STEPS  = 100
	FUTURE_STEPS = 100
	D            = 2
	NOISE        = 1

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
		print('Generating Spiral dataset...')
		data = self._generate_random_trajectories(self.n_training_samples)
		torch.save(data, os.path.join(self.data_folder, self.training_file))
		print('Done.')

	def _generate_random_trajectories(self, n_samples):
		batch = generate_spiral2d_matrix(
			n_spirals=n_samples,
			train_steps=self.TRAIN_STEPS,
			future_steps=self.FUTURE_STEPS,
			noise_std=0.0,   # шум добавляем отдельно через noise_weight после нормализации
			noise_mode='none',
			seed=42,
		)
		# full_trajs: (n_samples, T, 2)
		return batch.full_trajs.numpy().astype(np.float32)

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
