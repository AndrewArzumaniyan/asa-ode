import unittest
from types import SimpleNamespace

import torch
from torch.distributions.normal import Normal

from lib.create_latent_ode_model import create_LatentODE_model
from lib.feature_attn_latent_ode import (
	FeatureAttentionODEFunc,
	FeatureWiseDecoder,
	FeatureWiseEncoder_z0_RNN,
)


class FeatureAttentionLatentODETests(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(0)
		self.device = torch.device("cpu")
		self.batch_size = 3
		self.n_timepoints = 6
		self.n_features = 4
		self.feature_latent_dim = 5
		self.feature_embed_dim = 7
		self.total_latent_dim = self.n_features * self.feature_latent_dim

		self.time_steps = torch.linspace(0, 1, self.n_timepoints)
		self.truth = torch.randn(self.batch_size, self.n_timepoints, self.n_features)
		self.mask = torch.ones_like(self.truth)
		self.truth_w_mask = torch.cat((self.truth, self.mask), dim=-1)

	def test_feature_wise_encoder_returns_flattened_gaussian_params(self):
		encoder = FeatureWiseEncoder_z0_RNN(
			n_features=self.n_features,
			feature_latent_dim=self.feature_latent_dim,
			feature_embed_dim=self.feature_embed_dim,
			encoder_hidden_dim=9,
			device=self.device,
		)

		mean, std = encoder(self.truth_w_mask, self.time_steps)

		self.assertEqual(mean.shape, (1, self.batch_size, self.total_latent_dim))
		self.assertEqual(std.shape, (1, self.batch_size, self.total_latent_dim))
		self.assertTrue(torch.all(std >= 0))

	def test_feature_attention_ode_func_preserves_state_shape(self):
		ode_func = FeatureAttentionODEFunc(
			n_features=self.n_features,
			feature_latent_dim=self.feature_latent_dim,
			feature_embed_dim=self.feature_embed_dim,
			n_heads=1,
			n_layers=2,
		)

		state = torch.randn(2, self.batch_size, self.total_latent_dim)
		gradient = ode_func(torch.tensor(0.0), state)

		self.assertEqual(gradient.shape, state.shape)

	def test_feature_attention_ode_func_exposes_attention_maps(self):
		ode_func = FeatureAttentionODEFunc(
			n_features=self.n_features,
			feature_latent_dim=self.feature_latent_dim,
			feature_embed_dim=self.feature_embed_dim,
			n_heads=1,
			n_layers=2,
		)

		state = torch.randn(2, self.batch_size, self.total_latent_dim)
		attention_maps = ode_func.get_attention_maps(state)

		self.assertEqual(len(attention_maps), 2)
		for attn_map in attention_maps:
			self.assertEqual(
				attn_map.shape,
				(2, self.batch_size, 1, self.n_features, self.n_features),
			)
			self.assertTrue(
				torch.allclose(
					attn_map.sum(-1),
					torch.ones_like(attn_map.sum(-1)),
					atol=1e-5,
					rtol=1e-4,
				)
			)

	def test_feature_wise_decoder_projects_back_to_feature_space(self):
		decoder = FeatureWiseDecoder(
			n_features=self.n_features,
			feature_latent_dim=self.feature_latent_dim,
			feature_embed_dim=self.feature_embed_dim,
			decoder_hidden_dim=11,
		)

		latent_traj = torch.randn(
			2, self.batch_size, self.n_timepoints, self.total_latent_dim
		)
		reconstruction = decoder(latent_traj)

		self.assertEqual(
			reconstruction.shape,
			(2, self.batch_size, self.n_timepoints, self.n_features),
		)

	def test_factory_builds_feature_attention_latent_ode_and_reconstructs(self):
		args = SimpleNamespace(
			feature_attn_ode=True,
			feature_latents=self.feature_latent_dim,
			feature_embed_dim=self.feature_embed_dim,
			attn_heads=1,
			attn_layers=2,
			attn_dropout=0.0,
			decoder_units=13,
			poisson=False,
			classif=False,
			linear_classif=False,
			dataset="hopper",
			gen_layers=1,
			rec_layers=1,
			units=16,
			gru_units=12,
			rec_dims=9,
			z0_encoder="rnn",
			latents=6,
		)

		obsrv_std = torch.tensor([0.01])
		z0_prior = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
		model = create_LatentODE_model(
			args,
			self.n_features,
			z0_prior,
			obsrv_std,
			self.device,
		)

		time_steps_to_predict = torch.linspace(1.1, 1.5, 4)
		pred_x, info = model.get_reconstruction(
			time_steps_to_predict=time_steps_to_predict,
			truth=self.truth,
			truth_time_steps=self.time_steps,
			mask=self.mask,
			n_traj_samples=2,
		)

		self.assertEqual(
			pred_x.shape,
			(2, self.batch_size, len(time_steps_to_predict), self.n_features),
		)
		first_point_mu, first_point_std, first_point_enc = info["first_point"]
		self.assertEqual(
			first_point_mu.shape,
			(1, self.batch_size, self.total_latent_dim),
		)
		self.assertEqual(
			first_point_std.shape,
			(1, self.batch_size, self.total_latent_dim),
		)
		self.assertEqual(
			first_point_enc.shape,
			(2, self.batch_size, self.total_latent_dim),
		)


if __name__ == "__main__":
	unittest.main()
