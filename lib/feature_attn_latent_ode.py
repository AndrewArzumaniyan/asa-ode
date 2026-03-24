import torch
import torch.nn as nn

import lib.utils as utils
from lib.encoder_decoder import Encoder_z0_RNN


class FeatureWiseEncoder_z0_RNN(nn.Module):
	def __init__(
		self,
		n_features,
		feature_latent_dim,
		feature_embed_dim,
		encoder_hidden_dim = 20,
		use_delta_t = True,
		feature_embedding = None,
		device = torch.device("cpu"),
	):
		super(FeatureWiseEncoder_z0_RNN, self).__init__()

		self.n_features = n_features
		self.feature_latent_dim = feature_latent_dim
		self.total_latent_dim = n_features * feature_latent_dim
		self.device = device

		if feature_embedding is None:
			self.feature_embedding = nn.Embedding(n_features, feature_embed_dim)
			nn.init.normal_(self.feature_embedding.weight, mean=0, std=0.1)
		else:
			self.feature_embedding = feature_embedding

		self.register_buffer("feature_indices", torch.arange(n_features).long())

		self.encoder = Encoder_z0_RNN(
			latent_dim = feature_latent_dim,
			input_dim = 2 + feature_embed_dim,
			lstm_output_size = encoder_hidden_dim,
			use_delta_t = use_delta_t,
			device = device,
		).to(device)

	def forward(self, data, time_steps, run_backwards = True):
		n_traj, n_tp, n_dims = data.size()
		if n_dims != self.n_features * 2:
			raise ValueError(
				"Expected concatenated values and masks with last dim {}, got {}".format(
					self.n_features * 2, n_dims
				)
			)

		values = data[:, :, :self.n_features]
		mask = data[:, :, self.n_features:]
		utils.check_mask(values, mask)

		feature_embeddings = self.feature_embedding(self.feature_indices)
		feature_embeddings = feature_embeddings.view(1, 1, self.n_features, -1)
		feature_embeddings = feature_embeddings.expand(n_traj, n_tp, self.n_features, -1)

		feature_inputs = torch.cat(
			(
				values.unsqueeze(-1),
				mask.unsqueeze(-1),
				feature_embeddings,
			),
			-1,
		)

		feature_inputs = feature_inputs.permute(0, 2, 1, 3).reshape(
			n_traj * self.n_features, n_tp, -1
		)

		mean, std = self.encoder(
			feature_inputs,
			time_steps,
			run_backwards = run_backwards,
		)

		mean = mean.reshape(1, n_traj, self.n_features, self.feature_latent_dim)
		std = std.reshape(1, n_traj, self.n_features, self.feature_latent_dim)

		mean = mean.reshape(1, n_traj, self.total_latent_dim)
		std = std.reshape(1, n_traj, self.total_latent_dim)
		return mean, std


class FeatureSelfAttentionBlock(nn.Module):
	def __init__(self, embed_dim, n_heads, ff_hidden_dim, dropout = 0.0):
		super(FeatureSelfAttentionBlock, self).__init__()

		self.attention = nn.MultiheadAttention(
			embed_dim = embed_dim,
			num_heads = n_heads,
			dropout = dropout,
			batch_first = True,
		)
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)
		self.ff = nn.Sequential(
			nn.Linear(embed_dim, ff_hidden_dim),
			nn.Tanh(),
			nn.Linear(ff_hidden_dim, embed_dim),
		)
		utils.init_network_weights(self.ff)

	def forward(self, state):
		attended, _ = self.attention(state, state, state)
		state = self.norm1(state + attended)
		ff_state = self.ff(state)
		return self.norm2(state + ff_state)


class FeatureAttentionODEFunc(nn.Module):
	def __init__(
		self,
		n_features,
		feature_latent_dim,
		feature_embed_dim,
		n_heads = 1,
		n_layers = 1,
		n_units = 100,
		dropout = 0.0,
		feature_embedding = None,
		device = torch.device("cpu"),
	):
		super(FeatureAttentionODEFunc, self).__init__()

		if feature_latent_dim % n_heads != 0:
			raise ValueError("feature_latent_dim must be divisible by n_heads")

		self.n_features = n_features
		self.feature_latent_dim = feature_latent_dim
		self.total_latent_dim = n_features * feature_latent_dim
		self.device = device

		if feature_embedding is None:
			self.feature_embedding = nn.Embedding(n_features, feature_embed_dim)
			nn.init.normal_(self.feature_embedding.weight, mean=0, std=0.1)
		else:
			self.feature_embedding = feature_embedding

		self.register_buffer("feature_indices", torch.arange(n_features).long())

		self.feature_to_latent = nn.Linear(feature_embed_dim, feature_latent_dim)
		self.attn_blocks = nn.ModuleList(
			[
				FeatureSelfAttentionBlock(
					embed_dim = feature_latent_dim,
					n_heads = n_heads,
					ff_hidden_dim = n_units,
					dropout = dropout,
				)
				for _ in range(n_layers)
			]
		)
		self.gradient_net = nn.Sequential(
			nn.Linear(feature_latent_dim * 2, n_units),
			nn.Tanh(),
			nn.Linear(n_units, feature_latent_dim),
		)
		utils.init_network_weights(self.feature_to_latent)
		utils.init_network_weights(self.gradient_net)

	def _reshape_state(self, state):
		if state.size(-1) != self.total_latent_dim:
			raise ValueError(
				"Expected last dim {}, got {}".format(
					self.total_latent_dim, state.size(-1)
				)
			)

		leading_shape = state.size()[:-1]
		state = state.reshape(-1, self.n_features, self.feature_latent_dim)
		return state, leading_shape

	def get_ode_gradient_nn(self, state):
		state, leading_shape = self._reshape_state(state)

		feature_embeddings = self.feature_embedding(self.feature_indices)
		feature_bias = self.feature_to_latent(feature_embeddings)
		feature_bias = feature_bias.unsqueeze(0).expand(state.size(0), -1, -1)

		hidden = state + feature_bias
		for block in self.attn_blocks:
			hidden = block(hidden)

		gradient = self.gradient_net(torch.cat((hidden, feature_bias), -1))
		return gradient.reshape(*leading_shape, self.total_latent_dim)

	def forward(self, t_local, state, backwards = False):
		gradient = self.get_ode_gradient_nn(state)
		if backwards:
			gradient = -gradient
		return gradient

	def sample_next_point_from_prior(self, t_local, state):
		return self.get_ode_gradient_nn(state)


class FeatureWiseDecoder(nn.Module):
	def __init__(
		self,
		n_features,
		feature_latent_dim,
		feature_embed_dim,
		decoder_hidden_dim = 100,
		feature_embedding = None,
	):
		super(FeatureWiseDecoder, self).__init__()

		self.n_features = n_features
		self.feature_latent_dim = feature_latent_dim
		self.total_latent_dim = n_features * feature_latent_dim

		if feature_embedding is None:
			self.feature_embedding = nn.Embedding(n_features, feature_embed_dim)
			nn.init.normal_(self.feature_embedding.weight, mean=0, std=0.1)
		else:
			self.feature_embedding = feature_embedding

		self.register_buffer("feature_indices", torch.arange(n_features).long())
		self.decoder = nn.Sequential(
			nn.Linear(feature_latent_dim + feature_embed_dim, decoder_hidden_dim),
			nn.Tanh(),
			nn.Linear(decoder_hidden_dim, 1),
		)
		utils.init_network_weights(self.decoder)

	def forward(self, data):
		if data.size(-1) != self.total_latent_dim:
			raise ValueError(
				"Expected last dim {}, got {}".format(
					self.total_latent_dim, data.size(-1)
				)
			)

		leading_shape = data.size()[:-1]
		data = data.reshape(-1, self.n_features, self.feature_latent_dim)

		feature_embeddings = self.feature_embedding(self.feature_indices)
		feature_embeddings = feature_embeddings.unsqueeze(0).expand(data.size(0), -1, -1)

		decoded = self.decoder(torch.cat((data, feature_embeddings), -1)).squeeze(-1)
		return decoded.reshape(*leading_shape, self.n_features)
