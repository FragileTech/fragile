import torch

from fragile.core.layers import (
    PrimitiveAttentiveAtlasEncoder,
    PrimitiveTopologicalDecoder,
    TopoEncoderPrimitives,
)


def test_primitive_attentive_atlas_encoder_shapes() -> None:
    torch.manual_seed(0)
    encoder = PrimitiveAttentiveAtlasEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(4, 3)
    (
        K_chart,
        K_code,
        z_n,
        z_tex,
        router_weights,
        z_geo,
        vq_loss,
        indices_stack,
        z_n_all_charts,
    ) = encoder(x)

    assert K_chart.shape == (4,)
    assert K_code.shape == (4,)
    assert z_n.shape == (4, 2)
    assert z_tex.shape == (4, 2)
    assert router_weights.shape == (4, 3)
    assert z_geo.shape == (4, 2)
    assert vq_loss.ndim == 0
    assert indices_stack.shape == (4, 3)
    assert z_n_all_charts.shape == (4, 3, 2)


def test_primitive_topological_decoder_shapes() -> None:
    torch.manual_seed(1)
    decoder = PrimitiveTopologicalDecoder(
        latent_dim=2,
        hidden_dim=32,
        num_charts=3,
        output_dim=3,
    )
    z_geo = torch.randn(4, 2)
    z_tex = torch.randn(4, 2)
    x_hat, router_weights = decoder(z_geo, z_tex)

    assert x_hat.shape == (4, 3)
    assert router_weights.shape == (4, 3)
    assert torch.allclose(router_weights.sum(dim=-1), torch.ones(4), atol=1e-5)


def test_topoencoder_primitives_forward_and_losses() -> None:
    torch.manual_seed(2)
    model = TopoEncoderPrimitives(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(5, 3)
    x_recon, vq_loss, enc_weights, dec_weights, K_chart = model(x)

    assert x_recon.shape == (5, 3)
    assert vq_loss.ndim == 0
    assert enc_weights.shape == (5, 3)
    assert dec_weights.shape == (5, 3)
    assert K_chart.shape == (5,)

    consistency = model.compute_consistency_loss(enc_weights, dec_weights)
    assert consistency.ndim == 0
    assert model.compute_perplexity(K_chart) > 0.0
