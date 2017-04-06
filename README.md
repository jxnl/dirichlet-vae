# Dirichlet Variational Autoencoder

Variational Autoencoders (VAE) are extremely appealing models that allow for learning complicated distributions by taking advantage of recent progress in gradient descent algorithms and accelerated processing with GPUs. We modify the Stocastic Gradient Variational Bayes to perform posterior inference for latent spaces of multinomial distributions.
We propose Dirichlet priors for a multinomial latent distribution which allows us to explore the data by interpreting the discrete probabilies as class attributes. We emperically show that our Dirichlet model learns superior representations that outperform simpler Logistic Normal models for nearest neighbours, and have similar performance in supervised settings with the Normal VAEs.

This work was inspired by recent papers published in the International Conference on Learning Representations. Done as part of the final project of STAT 440: Computational Inference taught by Martin Lysy.
