# 🎨 Mixture-Density-Nets
A small PyTorch library for Mixture Density Networks.

# Install
simply run 
``pip install mixture-density-nets``

# Example
```py
from mixture_density_nets import MDN, MDDistribution
# ....
mdn = MDN(in_dim, out_dim, n_components)
# ....
mu, sigma, lambda_ = mdn(net(input_data))
dist = MDDistribution(mu, sigma, lambda_)
loss = dist.nll(targets).mean()

# ...
samples, clusters = dist.sample(n=20)  # draw 20 samples
```
For a more thorough example see [example.ipynb](example.ipynb).
