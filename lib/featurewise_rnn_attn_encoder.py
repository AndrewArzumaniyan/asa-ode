import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.utils as utils


def compute_feature_means(values, mask, global_feature_means = None):
	if values.shape != mask.shape:
		raise ValueError(
			"values and mask must have identical shapes, got {} and {}".format(
				values.shape, mask.shape
			)
		)

	if values.dim() != 3:
		raise ValueError("Expected [batch, time, features] tensor, got {}".format(values.shape))

	sums = (values * mask).sum(dim = 1)
	counts = mask.sum(dim = 1)

	if global_feature_means is None:
		global_feature_means = torch.zeros(values.size(-1), device = values.device, dtype = values.dtype)

	global_feature_means = global_feature_means.to(device = values.device, dtype = values.dtype)
	global_feature_means = global_feature_means.unsqueeze(0).expand(values.size(0), -1)

	means = sums / counts.clamp_min(1.0)
	means = torch.where(counts > 0, means, global_feature_means)
	return means


class FeatureScalarEmbedding(nn.Module):
	def __init__(
		self,
		n_features,
		embed_dim,
		global_feature_means = None,
		feature_embedding = None,
	):
		super(FeatureScalarEmbedding, self).__init__()

		self.n_features = n_features
		self.embed_dim = embed_dim

		if feature_embedding is None:
			self.feature_embedding = nn.Embedding(n_features, embed_dim)
			nn.init.normal_(self.feature_embedding.weight, mean = 0.0, std = 0.1)
		else:
			self.feature_embedding = feature_embedding

		self.value_projection = nn.Parameter(torch.randn(embed_dim) * 0.1)
		self.register_buffer("feature_indices", torch.arange(n_features).long())

		if global_feature_means is None:
			global_feature_means = torch.zeros(n_features)
		self.register_buffer("global_feature_means", global_feature_means.float())

	def forward(self, values_t, mask_t, previous_attention = None, sequence_feature_means = None):
		if values_t.dim() != 2 or mask_t.dim() != 2:
			raise ValueError("Expected [batch, features] values and mask")

		if values_t.shape != mask_t.shape:
			raise ValueError(
				"values_t and mask_t must match, got {} and {}".format(
					values_t.shape, mask_t.shape
				)
			)

		if values_t.size(-1) != self.n_features:
			raise ValueError(
				"Expected {} features, got {}".format(self.n_features, values_t.size(-1))
			)

		feature_bias = self.feature_embedding(self.feature_indices)
		feature_bias = feature_bias.unsqueeze(0).expand(values_t.size(0), -1, -1)
		projected = values_t.unsqueeze(-1) * self.value_projection.view(1, 1, -1)
		observed_embeddings = projected + feature_bias

		if previous_attention is not None:
			if previous_attention.shape != observed_embeddings.shape:
				raise ValueError(
					"previous_attention must have shape {}, got {}".format(
						observed_embeddings.shape, previous_attention.shape
					)
				)
			missing_embeddings = previous_attention
		else:
			if sequence_feature_means is None:
				sequence_feature_means = self.global_feature_means.unsqueeze(0).expand(values_t.size(0), -1)

			if sequence_feature_means.shape != values_t.shape:
				raise ValueError(
					"sequence_feature_means must have shape {}, got {}".format(
						values_t.shape, sequence_feature_means.shape
					)
				)

			missing_embeddings = (
				sequence_feature_means.unsqueeze(-1) * self.value_projection.view(1, 1, -1)
				+ feature_bias
			)

		mask = mask_t.unsqueeze(-1).bool()
		return torch.where(mask, observed_embeddings, missing_embeddings)


class MaskedFeatureAttention(nn.Module):
	def __init__(self, embed_dim, n_heads = 1):
		super(MaskedFeatureAttention, self).__init__()

		if embed_dim % n_heads != 0:
			raise ValueError("embed_dim must be divisible by n_heads")

		self.embed_dim = embed_dim
		self.n_heads = n_heads
		self.head_dim = embed_dim // n_heads

		self.q_proj = nn.Linear(embed_dim, embed_dim, bias = False)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias = False)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias = False)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias = False)

	def _reshape_heads(self, tensor):
		batch_size, n_features, _ = tensor.shape
		tensor = tensor.reshape(batch_size, n_features, self.n_heads, self.head_dim)
		return tensor.permute(0, 2, 1, 3)

	def forward(self, embeddings, observed_mask, return_weights = False):
		if embeddings.dim() != 3:
			raise ValueError("Expected [batch, features, embed_dim], got {}".format(embeddings.shape))

		if observed_mask.dim() != 2:
			raise ValueError("Expected [batch, features] observed_mask")

		if embeddings.size(0) != observed_mask.size(0) or embeddings.size(1) != observed_mask.size(1):
			raise ValueError(
				"embeddings and observed_mask batch/feature dims must match, got {} and {}".format(
					embeddings.shape, observed_mask.shape
				)
			)

		queries = self._reshape_heads(self.q_proj(embeddings))
		keys = self._reshape_heads(self.k_proj(embeddings))
		values = self._reshape_heads(self.v_proj(embeddings))

		scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
		key_mask = observed_mask.bool().unsqueeze(1).unsqueeze(2)
		scores = scores.masked_fill(~key_mask, float("-inf"))

		weights = torch.zeros_like(scores)
		has_observed = observed_mask.sum(-1) > 0
		if has_observed.any():
			weights[has_observed] = torch.softmax(scores[has_observed], dim = -1)

		context = torch.matmul(weights, values)
		context = context.permute(0, 2, 1, 3).reshape(embeddings.size(0), embeddings.size(1), self.embed_dim)
		attended = self.out_proj(context)

		if return_weights:
			return attended, weights
		return attended


class FeatureWiseLinearDecoder(nn.Module):
	def __init__(self, n_features, feature_latent_dim):
		super(FeatureWiseLinearDecoder, self).__init__()

		self.n_features = n_features
		self.feature_latent_dim = feature_latent_dim
		self.total_latent_dim = n_features * feature_latent_dim
		self.projection = nn.Linear(feature_latent_dim, 1)

	def forward(self, latent_traj):
		if latent_traj.size(-1) != self.total_latent_dim:
			raise ValueError(
				"Expected last dim {}, got {}".format(
					self.total_latent_dim, latent_traj.size(-1)
				)
			)

		leading_shape = latent_traj.size()[:-1]
		latent_traj = latent_traj.reshape(-1, self.n_features, self.feature_latent_dim)
		decoded = self.projection(latent_traj).squeeze(-1)
		return decoded.reshape(*leading_shape, self.n_features)


class FeatureWiseRNNEncoder(nn.Module):
	def __init__(
		self,
		n_features,
		feature_latent_dim,
		feature_embed_dim,
		encoder_hidden_dim = 20,
		n_heads = 1,
		n_attention_layers = 1,
		global_feature_means = None,
		feature_embedding = None,
		device = torch.device("cpu"),
	):
		super(FeatureWiseRNNEncoder, self).__init__()

		self.n_features = n_features
		self.feature_latent_dim = feature_latent_dim
		self.feature_embed_dim = feature_embed_dim
		self.encoder_hidden_dim = encoder_hidden_dim
		self.total_latent_dim = n_features * feature_latent_dim
		self.device = device

		self.embedding = FeatureScalarEmbedding(
			n_features = n_features,
			embed_dim = feature_embed_dim,
			global_feature_means = global_feature_means,
			feature_embedding = feature_embedding,
		)
		self.attention_layers = nn.ModuleList(
			[MaskedFeatureAttention(feature_embed_dim, n_heads = n_heads) for _ in range(n_attention_layers)]
		)
		self.gru_cell = nn.GRUCell(feature_embed_dim * 2 + 1, encoder_hidden_dim)
		self.hidden_to_mean = nn.Linear(encoder_hidden_dim, feature_latent_dim)
		self.hidden_to_std = nn.Linear(encoder_hidden_dim, feature_latent_dim)
		self.extra_info = None

	def _split_values_and_mask(self, data):
		if data.size(-1) == self.n_features:
			values = data
			mask = torch.ones_like(values)
			return values, mask

		if data.size(-1) == self.n_features * 2:
			values = data[:, :, :self.n_features]
			mask = data[:, :, self.n_features:]
			utils.check_mask(values, mask)
			return values, mask

		raise ValueError(
			"Expected last dim {} or {}, got {}".format(
				self.n_features, self.n_features * 2, data.size(-1)
			)
		)

	def _get_processing_order(self, time_steps, run_backwards):
		if run_backwards:
			return list(range(len(time_steps) - 1, -1, -1))
		return list(range(len(time_steps)))

	def _get_delta_ts(self, time_steps, run_backwards):
		if len(time_steps) == 1:
			return torch.zeros(1, device = time_steps.device, dtype = time_steps.dtype)

		delta_ts = time_steps[1:] - time_steps[:-1]
		if torch.any(delta_ts <= 0):
			raise ValueError("time_steps must be strictly increasing")

		zero = torch.zeros(1, device = time_steps.device, dtype = time_steps.dtype)
		if run_backwards:
			return torch.cat((zero, torch.flip(delta_ts, dims = [0])), dim = 0)
		return torch.cat((zero, delta_ts), dim = 0)

	def _get_feature_identity(self, batch_size, device, dtype):
		feature_identity = self.embedding.feature_embedding(self.embedding.feature_indices)
		feature_identity = feature_identity.to(device = device, dtype = dtype)
		return feature_identity.unsqueeze(0).expand(batch_size, -1, -1)

	def forward(self, data, time_steps, run_backwards = True):
		if data.dim() != 3:
			raise ValueError("Expected [batch, time, features] input, got {}".format(data.shape))

		values, mask = self._split_values_and_mask(data)
		sequence_feature_means = compute_feature_means(
			values,
			mask,
			self.embedding.global_feature_means,
		)

		batch_size, n_timepoints, _ = values.shape
		processing_order = self._get_processing_order(time_steps, run_backwards)
		delta_ts = self._get_delta_ts(time_steps, run_backwards)

		hidden = torch.zeros(
			batch_size,
			self.n_features,
			self.encoder_hidden_dim,
			device = values.device,
			dtype = values.dtype,
		)
		prev_attention = None
		feature_identity = self._get_feature_identity(batch_size, values.device, values.dtype)

		attention_trace = []
		hidden_trace = []
		mask_trace = []

		for step_idx, time_index in enumerate(processing_order):
			values_t = values[:, time_index, :]
			mask_t = mask[:, time_index, :]
			has_observed = mask_t.sum(-1) > 0

			embeddings = self.embedding(
				values_t,
				mask_t,
				previous_attention = prev_attention,
				sequence_feature_means = sequence_feature_means,
			)

			if prev_attention is None:
				attention_output = embeddings.clone()
			else:
				attention_output = prev_attention.clone()

			if has_observed.any():
				valid_embeddings = embeddings[has_observed]
				valid_mask = mask_t[has_observed]
				for attention_layer in self.attention_layers:
					valid_embeddings = attention_layer(valid_embeddings, valid_mask)
				attention_output[has_observed] = valid_embeddings

				delta_input = torch.full(
					(valid_embeddings.size(0), self.n_features, 1),
					delta_ts[step_idx].item(),
					device = values.device,
					dtype = values.dtype,
				)
				gru_input = torch.cat((valid_embeddings, feature_identity[has_observed], delta_input), dim = -1)
				updated_hidden = self.gru_cell(
					gru_input.reshape(-1, gru_input.size(-1)),
					hidden[has_observed].reshape(-1, self.encoder_hidden_dim),
				)
				hidden[has_observed] = updated_hidden.reshape(-1, self.n_features, self.encoder_hidden_dim)

			prev_attention = attention_output
			attention_trace.append(attention_output.detach())
			hidden_trace.append(hidden.detach())
			mask_trace.append(mask_t.detach())

		final_hidden = hidden
		mean = self.hidden_to_mean(final_hidden)
		std = F.softplus(self.hidden_to_std(final_hidden)) + 1e-5

		self.extra_info = {
			"sequence_feature_means": sequence_feature_means.detach(),
			"processed_indices": torch.tensor(processing_order, device = time_steps.device),
			"delta_ts": delta_ts.detach(),
			"attention_outputs": torch.stack(attention_trace, dim = 1),
			"hidden_states": torch.stack(hidden_trace, dim = 1),
			"processed_masks": torch.stack(mask_trace, dim = 1),
		}

		mean = mean.reshape(batch_size, self.total_latent_dim).unsqueeze(0)
		std = std.reshape(batch_size, self.total_latent_dim).unsqueeze(0)
		return mean, std
