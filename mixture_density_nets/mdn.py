import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class MDDistribution:
    def __init__(self, mu, sigma, lambda_logits):
        super().__init__()
        self.normal_dist = Normal(mu, sigma)
        self.lambda_logits = lambda_logits

    def log_prob(self, y):
        lp_normals = self.normal_dist.log_prob(y.unsqueeze(-2))
        lp_lambdas = self.lambda_logits.log_softmax(-2)
        log_prob = torch.logsumexp(lp_lambdas + lp_normals, dim=-2)
        return log_prob

    def nll(self, y):
        return -self.log_prob(y)

    @torch.inference_mode(True)
    def sample(self, n=1):
        categorical_sample = F.gumbel_softmax(
            self.lambda_logits.expand(n, *[-1]*(len(self.lambda_logits.shape))),
            hard=True, tau=1, dim=-2
        )
        normal_samples = self.normal_dist.sample((n, ))
        samples = (categorical_sample * normal_samples).sum(-2).squeeze()
        clusters = categorical_sample.squeeze().argmax(-1)
        return samples.transpose(0, 1), clusters.transpose(0, 1)


class MDN(nn.Module):
    def __init__(self, in_dim, out_dim, n_components):
        super().__init__()
        # bookkeeping
        self.out_dim = out_dim
        self.n_components = n_components
        # distributions
        self.sigma_head  = nn.Linear(in_dim, n_components * out_dim)
        self.mu_head     = nn.Linear(in_dim, n_components * out_dim)
        self.lambda_head = nn.Linear(in_dim, n_components)
        # helper
        self.unflatten   = nn.Unflatten(-1, (n_components, out_dim))

    def forward(self, x):
        mu = self.unflatten(self.mu_head(x))
        sigma = self.unflatten(F.softplus(self.sigma_head(x)) + 1e-12)
        lambda_logits = self.lambda_head(x)
        return mu, sigma, lambda_logits.unsqueeze(-1)


# aliases
MixtureDensityNet = MDN
MixtureDensityDistribution = MDDistribution
