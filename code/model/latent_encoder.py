from model.aggregator import AverageAggregator
from model.transformer import *


class LatentEncoder(torch.nn.Module):
    """The Latent Encoder."""

    def __init__(self, d_model, num_hidden, num_latent, nhead, dim_feedforward=2048, dropout=0.1):
        super(LatentEncoder, self).__init__()
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
        self._num_latent = num_latent
        self.aggregator = AverageAggregator()

        self.mu = Linear(num_hidden, num_latent)
        self.log_sigma = Linear(num_hidden, num_latent)

    def forward(self, src, src_mask=None):
        latent_representations = self.encoder(src=src, src_key_padding_mask=src_mask)

        latent_aggregated_representation = self.aggregator(x=latent_representations.transpose(0, 1), x_mask=src_mask)

        # get mu and sigma
        mu = self.mu(latent_aggregated_representation)
        log_sigma = self.log_sigma(latent_aggregated_representation)

        # reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return mu, log_sigma, z

