from .transformer import *
from .aggregator import AverageAggregator

class LatentEncoder(torch.nn.Module):
    """The Latent Encoder."""

    def __init__(self, d_model, num_latents, nhead, dim_feedforward=2048, dropout=0.1):
        """(A)NP latent encoder.

        Args:
            d_model: An iterable containing the output sizes of the encoding MLP.
            num_latents: The latent dimensionality.
            nhead: Number of attention heads.
            dim_feedforward: Dimension of feedforward layer.
            dropout: Dropout ratio.
        """
        self.encoder = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(d_model=d_model,
                                                  nhead=nhead,
                                                  dim_feedforward=dim_feedforward,
                                                  dropout=dropout),
            num_layers=6
        )
        self._num_latents = num_latents
        self.aggregator = AverageAggregator()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        latent_representations = self.encoder(src=src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

        latent_aggregated_representation = self.aggregator(x=latent_representations, x_mask=src_mask)

        return latent_aggregated_representation

