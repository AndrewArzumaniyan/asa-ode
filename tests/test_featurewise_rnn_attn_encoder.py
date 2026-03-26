import math
import unittest
from types import SimpleNamespace

import torch
from torch.distributions.normal import Normal

from lib.create_latent_ode_model import create_LatentODE_model
from lib.featurewise_rnn_attn_encoder import (
	FeatureScalarEmbedding,
	FeatureWiseLinearDecoder,
	FeatureWiseRNNEncoder,
	MaskedFeatureAttention,
	compute_feature_means,
)
from lib.ode_func import ODEFunc


class FeatureWiseEncoderBuildingBlockTests(unittest.TestCase):
	def test_compute_feature_means_uses_global_fallback_for_unobserved_features(self):
		values = torch.tensor(
			[
				[
					[1.0, 0.0, 5.0],
					[3.0, 0.0, 0.0],
					[0.0, 0.0, 7.0],
				]
			]
		)
		mask = torch.tensor(
			[
				[
					[1.0, 0.0, 1.0],
					[1.0, 0.0, 0.0],
					[0.0, 0.0, 1.0],
				]
			]
		)
		global_feature_means = torch.tensor([10.0, 20.0, 30.0])

		means = compute_feature_means(values, mask, global_feature_means)

		expected = torch.tensor([[2.0, 20.0, 6.0]])
		self.assertTrue(torch.allclose(means, expected))

	def test_feature_scalar_embedding_uses_previous_attention_for_missing_features(self):
		embedding = FeatureScalarEmbedding(
			n_features=3,
			embed_dim=2,
			global_feature_means=torch.tensor([0.0, 0.0, 0.0]),
		)

		with torch.no_grad():
			embedding.value_projection.copy_(torch.tensor([2.0, -1.0]))
			embedding.feature_embedding.weight.copy_(
				torch.tensor(
					[
						[1.0, 1.0],
						[10.0, 10.0],
						[100.0, 100.0],
					]
				)
			)

		values_t = torch.tensor([[0.5, 9.0, -3.0]])
		mask_t = torch.tensor([[1.0, 0.0, 1.0]])
		previous_attention = torch.tensor(
			[
				[
					[7.0, 8.0],
					[20.0, 21.0],
					[30.0, 31.0],
				]
			]
		)
		sequence_feature_means = torch.tensor([[0.25, 0.75, -1.0]])

		embedded = embedding(
			values_t,
			mask_t,
			previous_attention=previous_attention,
			sequence_feature_means=sequence_feature_means,
		)

		expected = torch.tensor(
			[
				[
					[2.0, 0.5],
					[20.0, 21.0],
					[94.0, 103.0],
				]
			]
		)
		self.assertTrue(torch.allclose(embedded, expected))

	def test_feature_scalar_embedding_uses_sequence_means_on_first_step(self):
		embedding = FeatureScalarEmbedding(
			n_features=3,
			embed_dim=2,
			global_feature_means=torch.tensor([0.0, 0.0, 0.0]),
		)

		with torch.no_grad():
			embedding.value_projection.copy_(torch.tensor([2.0, -1.0]))
			embedding.feature_embedding.weight.copy_(
				torch.tensor(
					[
						[1.0, 1.0],
						[10.0, 10.0],
						[100.0, 100.0],
					]
				)
			)

		values_t = torch.tensor([[0.5, 9.0, -3.0]])
		mask_t = torch.tensor([[1.0, 0.0, 1.0]])
		sequence_feature_means = torch.tensor([[0.25, 0.75, -1.0]])

		embedded = embedding(
			values_t,
			mask_t,
			previous_attention=None,
			sequence_feature_means=sequence_feature_means,
		)

		expected = torch.tensor(
			[
				[
					[2.0, 0.5],
					[11.5, 9.25],
					[94.0, 103.0],
				]
			]
		)
		self.assertTrue(torch.allclose(embedded, expected))

	def test_masked_feature_attention_masks_missing_keys(self):
		attention = MaskedFeatureAttention(embed_dim=2, n_heads=1)

		with torch.no_grad():
			identity = torch.eye(2)
			attention.q_proj.weight.copy_(identity)
			attention.k_proj.weight.copy_(identity)
			attention.v_proj.weight.copy_(identity)
			attention.out_proj.weight.copy_(identity)

		embeddings = torch.tensor(
			[
				[
					[1.0, 0.0],
					[0.0, 1.0],
					[0.0, 0.0],
				]
			]
		)
		observed_mask = torch.tensor([[1.0, 0.0, 1.0]])

		attended, weights = attention(
			embeddings,
			observed_mask,
			return_weights=True,
		)

		scale = math.sqrt(2.0)
		query0 = torch.softmax(torch.tensor([1.0 / scale, 0.0]), dim=0)
		expected_weights = torch.tensor(
			[
				[
					[
						[query0[0], 0.0, query0[1]],
						[0.5, 0.0, 0.5],
						[0.5, 0.0, 0.5],
					]
				]
			]
		)
		expected_attended = torch.tensor(
			[
				[
					[query0[0], 0.0],
					[0.5, 0.0],
					[0.5, 0.0],
				]
			]
		)

		self.assertTrue(torch.allclose(weights, expected_weights, atol=1e-6))
		self.assertTrue(torch.allclose(attended, expected_attended, atol=1e-6))

	def test_feature_wise_linear_decoder_applies_shared_projection(self):
		decoder = FeatureWiseLinearDecoder(n_features=3, feature_latent_dim=2)

		with torch.no_grad():
			decoder.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
			decoder.projection.bias.copy_(torch.tensor([0.5]))

		latent = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])
		reconstruction = decoder(latent)

		expected = torch.tensor([[[[0.5, 2.5, 4.5]]]])
		self.assertTrue(torch.allclose(reconstruction, expected))


class FeatureWiseRNNEncoderTests(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(0)

	def _make_encoder(self):
		encoder = FeatureWiseRNNEncoder(
			n_features=2,
			feature_latent_dim=1,
			feature_embed_dim=1,
			encoder_hidden_dim=1,
			n_heads=1,
			n_attention_layers=1,
			global_feature_means=torch.tensor([5.0, 7.0]),
		)

		with torch.no_grad():
			encoder.embedding.value_projection.copy_(torch.tensor([1.0]))
			encoder.embedding.feature_embedding.weight.zero_()

			attention = encoder.attention_layers[0]
			attention.q_proj.weight.fill_(1.0)
			attention.k_proj.weight.fill_(1.0)
			attention.v_proj.weight.fill_(1.0)
			attention.out_proj.weight.fill_(1.0)

			encoder.gru_cell.weight_ih.zero_()
			encoder.gru_cell.weight_hh.zero_()
			encoder.gru_cell.bias_ih.zero_()
			encoder.gru_cell.bias_hh.zero_()
			encoder.gru_cell.bias_ih[0].fill_(-10.0)
			encoder.gru_cell.bias_ih[1].fill_(-10.0)
			encoder.gru_cell.weight_ih[2, 2].fill_(1.0)

			encoder.hidden_to_mean.weight.fill_(1.0)
			encoder.hidden_to_mean.bias.zero_()
			encoder.hidden_to_std.weight.zero_()
			encoder.hidden_to_std.bias.zero_()

		return encoder

	def test_encoder_returns_flattened_gaussian_params_and_positive_std(self):
		encoder = self._make_encoder()
		time_steps = torch.tensor([0.0, 1.0, 2.0])
		values = torch.tensor(
			[
				[
					[1.0, 2.0],
					[0.0, 0.0],
					[3.0, 1.0],
				]
			]
		)
		mask = torch.tensor(
			[
				[
					[1.0, 1.0],
					[0.0, 0.0],
					[1.0, 1.0],
				]
			]
		)
		data = torch.cat((values, mask), dim=-1)

		mean, std = encoder(data, time_steps, run_backwards=True)

		self.assertEqual(mean.shape, (1, 1, 2))
		self.assertEqual(std.shape, (1, 1, 2))
		self.assertTrue(torch.all(std > 0))

	def test_encoder_uses_global_feature_means_for_features_never_observed(self):
		encoder = self._make_encoder()
		time_steps = torch.tensor([0.0, 1.0, 2.0])
		values = torch.tensor(
			[
				[
					[1.0, 0.0],
					[3.0, 0.0],
					[5.0, 0.0],
				]
			]
		)
		mask = torch.tensor(
			[
				[
					[1.0, 0.0],
					[1.0, 0.0],
					[1.0, 0.0],
				]
			]
		)
		data = torch.cat((values, mask), dim=-1)

		encoder(data, time_steps, run_backwards=True)

		expected_means = torch.tensor([[3.0, 7.0]])
		self.assertTrue(torch.allclose(encoder.extra_info["sequence_feature_means"], expected_means))

	def test_encoder_skips_attention_and_hidden_updates_on_all_missing_step(self):
		encoder = self._make_encoder()
		time_steps = torch.tensor([0.0, 1.0, 2.0])
		values = torch.tensor(
			[
				[
					[1.0, 2.0],
					[0.0, 0.0],
					[3.0, 1.0],
				]
			]
		)
		mask = torch.tensor(
			[
				[
					[1.0, 1.0],
					[0.0, 0.0],
					[1.0, 1.0],
				]
			]
		)
		data = torch.cat((values, mask), dim=-1)

		encoder(data, time_steps, run_backwards=True)

		attention_outputs = encoder.extra_info["attention_outputs"]
		hidden_states = encoder.extra_info["hidden_states"]

		self.assertTrue(torch.allclose(attention_outputs[:, 0], attention_outputs[:, 1]))
		self.assertFalse(torch.allclose(attention_outputs[:, 1], attention_outputs[:, 2]))
		self.assertTrue(torch.allclose(hidden_states[:, 0], hidden_states[:, 1]))

	def test_encoder_exposes_backward_delta_ts(self):
		encoder = self._make_encoder()
		time_steps = torch.tensor([0.0, 1.0, 2.0])
		values = torch.tensor(
			[
				[
					[1.0, 2.0],
					[0.0, 0.0],
					[3.0, 1.0],
				]
			]
		)
		mask = torch.tensor(
			[
				[
					[1.0, 1.0],
					[0.0, 0.0],
					[1.0, 1.0],
				]
			]
		)
		data = torch.cat((values, mask), dim=-1)

		encoder(data, time_steps, run_backwards=True)

		expected_delta_ts = torch.tensor([0.0, 1.0, 1.0])
		self.assertTrue(torch.allclose(encoder.extra_info["delta_ts"], expected_delta_ts))

	def test_encoder_assumes_all_features_observed_when_mask_is_absent(self):
		encoder = self._make_encoder()
		time_steps = torch.tensor([0.0, 1.0, 2.0])
		values = torch.tensor(
			[
				[
					[1.0, 2.0],
					[3.0, 4.0],
					[5.0, 6.0],
				]
			]
		)

		mean, std = encoder(values, time_steps, run_backwards=True)

		self.assertEqual(mean.shape, (1, 1, 2))
		self.assertEqual(std.shape, (1, 1, 2))
		self.assertTrue(torch.allclose(encoder.extra_info["sequence_feature_means"], torch.tensor([[3.0, 4.0]])))


class FeatureWiseLatentODEIntegrationTests(unittest.TestCase):
	def setUp(self):
		torch.manual_seed(0)
		self.device = torch.device("cpu")
		self.n_features = 3
		self.feature_latent_dim = 2
		self.total_latent_dim = self.n_features * self.feature_latent_dim
		self.time_steps = torch.tensor([0.0, 0.4, 0.8, 1.0])
		self.truth = torch.tensor(
			[
				[
					[0.1, 0.2, 0.3],
					[0.0, 0.4, 0.0],
					[0.5, 0.0, 0.7],
					[0.2, 0.6, 0.1],
				],
				[
					[0.3, 0.1, 0.2],
					[0.4, 0.0, 0.5],
					[0.0, 0.2, 0.6],
					[0.7, 0.8, 0.0],
				],
			]
		)
		self.mask = torch.tensor(
			[
				[
					[1.0, 1.0, 1.0],
					[0.0, 1.0, 0.0],
					[1.0, 0.0, 1.0],
					[1.0, 1.0, 1.0],
				],
				[
					[1.0, 1.0, 1.0],
					[1.0, 0.0, 1.0],
					[0.0, 1.0, 1.0],
					[1.0, 1.0, 0.0],
				],
			]
		)
		self.args = SimpleNamespace(
			feature_attn_ode=False,
			featurewise_rnn_attn_ode=True,
			feature_latents=self.feature_latent_dim,
			feature_embed_dim=4,
			attn_heads=1,
			attn_layers=1,
			attn_dropout=0.0,
			decoder_units=16,
			poisson=False,
			classif=False,
			linear_classif=False,
			dataset="hopper",
			gen_layers=1,
			rec_layers=1,
			units=12,
			gru_units=8,
			rec_dims=5,
			z0_encoder="rnn",
			latents=6,
		)

	def test_factory_builds_featurewise_encoder_with_standard_odefunc(self):
		model = create_LatentODE_model(
			self.args,
			self.n_features,
			Normal(torch.tensor([0.0]), torch.tensor([1.0])),
			torch.tensor([0.01]),
			self.device,
			global_feature_means=torch.tensor([0.2, 0.3, 0.4]),
		)

		self.assertIsInstance(model.encoder_z0, FeatureWiseRNNEncoder)
		self.assertIsInstance(model.decoder, FeatureWiseLinearDecoder)
		self.assertIsInstance(model.diffeq_solver.ode_func, ODEFunc)
		self.assertTrue(
			torch.allclose(
				model.encoder_z0.embedding.global_feature_means,
				torch.tensor([0.2, 0.3, 0.4]),
			)
		)

	def test_featurewise_model_reconstructs_and_computes_losses(self):
		model = create_LatentODE_model(
			self.args,
			self.n_features,
			Normal(torch.tensor([0.0]), torch.tensor([1.0])),
			torch.tensor([0.01]),
			self.device,
			global_feature_means=torch.tensor([0.2, 0.3, 0.4]),
		)

		pred_x, info = model.get_reconstruction(
			time_steps_to_predict=self.time_steps,
			truth=self.truth,
			truth_time_steps=self.time_steps,
			mask=self.mask,
			n_traj_samples=2,
		)

		self.assertEqual(pred_x.shape, (2, 2, len(self.time_steps), self.n_features))
		first_point_mu, first_point_std, first_point_enc = info["first_point"]
		self.assertEqual(first_point_mu.shape, (1, 2, self.total_latent_dim))
		self.assertEqual(first_point_std.shape, (1, 2, self.total_latent_dim))
		self.assertEqual(first_point_enc.shape, (2, 2, self.total_latent_dim))

		batch_dict = {
			"observed_data": self.truth,
			"observed_tp": self.time_steps,
			"data_to_predict": self.truth,
			"tp_to_predict": self.time_steps,
			"observed_mask": self.mask,
			"mask_predicted_data": self.mask,
			"labels": None,
			"mode": "interp",
		}
		losses = model.compute_all_losses(batch_dict, n_traj_samples=2, kl_coef=1.0)

		self.assertTrue(torch.isfinite(losses["loss"]))
		self.assertTrue(torch.isfinite(losses["likelihood"]))
		self.assertTrue(torch.isfinite(losses["mse"]))
		self.assertTrue(torch.isfinite(losses["kl_first_p"]))


if __name__ == "__main__":
	unittest.main()
