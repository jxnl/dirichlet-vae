import tensorflow as T
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics


class SamplingReparamKL:
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def to_mu(self, alpha):
        d1 = self.latent_dim - 1
        digamma_d = T.digamma(alpha[:, -1:])
        mu = T.digamma(alpha[:, :d1]) - digamma_d
        return mu

    def to_sd(self, alpha):
        _one = K.cast(1, dtype=alpha.dtype)
        d1 = self.latent_dim - 1
        var = (T.polygamma(_one, alpha[:, :d1])
               + T.polygamma(_one, alpha[:, -1:]))
        sigma = T.sqrt(var)
        return sigma

    def sample(self, args):
        mu, sigma = args
        e = K.random_normal(
            shape=(self.batch_size, self.latent_dim-1))
        z = mu + sigma * e
        one = K.ones((self.batch_size, 1))
        z_star = K.softmax(K.concatenate([z, one]))
        return z_star


class SamplingReparamLaplace:
    def __init__(self, latent_dim, batch_size):
        self.latent_dim = latent_dim
        self.batch_size = batch_size

    def to_mu(self, alpha):
        log_alpha = K.log(alpha)
        mean_log_alpha = K.mean(log_alpha, axis=-1, keepdims=True)
        mu = log_alpha - mean_log_alpha
        return mu

    def to_sd(self, alpha):
        k1 = 1 - (2 / self.latent_dim)
        k2 = 1 / (self.latent_dim ** 2)
        sigma = k1 * 1/alpha + k2 * K.sum(1/alpha, axis=-1, keepdims=True)
        return sigma

    def sample(self, args):
        mu, sigma = args
        e = K.random_normal(
            shape=(self.batch_size, self.latent_dim))
        z_star = K.softmax(mu + sigma * e)
        return z_star


class DirVae:
    def __init__(self,
                 original_dim,
                 reparam,
                 batch_size=16,
                 encoder_widths=50,
                 latent_dim=10,
                 decoder_width=50,
                 log_alpha=True,
                 dropout=True):

        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.original_dim = original_dim
        self.encoder_widths = encoder_widths
        self.latent_dim = latent_dim
        self.decoder_width = decoder_width
        self.reparam = reparam(self.latent_dim, self.batch_size)
        self.log_alpha = log_alpha
        self.dropout = dropout

        # Network Specification
        self.x = Input(batch_shape=(self.batch_size, original_dim))

        # Two-Layer Network learns q(z|x)
        # produces alpha(x)
        self.q_zx = Dense(self.encoder_widths, activation='relu')
        self.alpha = Dense(self.latent_dim, activation="tanh")
        self.scale = Dense(self.latent_dim, activation="softplus")

        # Two Layer Network learns p(x|z)
        # produces expectation
        self.p_xz = Dense(self.decoder_width, input_dim=self.latent_dim,
                          activation='relu')
        self.generator = Dense(self.original_dim, activation='sigmoid')

    def initialize(self):
        z = (self.q_zx(self.x))

        if self.dropout:
            z = Dropout(0.5)(z)

        alpha = self.alpha(z)
        scale = self.scale(z)

        def remapping(args):
            alpha, scale = args
            return K.exp(scale * alpha)

        alpha = Lambda(remapping,
                       output_shape=(self.latent_dim,))([alpha, scale])

        z_mu = Lambda(
            self.reparam.to_mu,
            output_shape=(self.latent_dim,))(alpha)

        z_sd = Lambda(
            self.reparam.to_sd,
            output_shape=(self.latent_dim,))(alpha)

        z_star = Lambda(
            self.reparam.sample,
            output_shape=(self.latent_dim,))([z_mu, z_sd])

        density = self.p_xz(z_star)
        x_star = self.generator(density)

        self.latent_variable = alpha

        def vae_loss(x, x_star):
            reconstruction_error = (
                self.original_dim * metrics.binary_crossentropy(x, x_star))

            a0 = K.sum(alpha, axis=-1, keepdims=True)
            kl = (T.lgamma(a0)
                  - K.sum(T.lgamma(alpha), axis=-1)
                  + K.sum(alpha * T.digamma(alpha), axis=-1)
                  - K.sum(alpha * T.digamma(a0), axis=-1)
                  - K.mean(T.digamma(alpha) - T.digamma(a0), axis=-1))
            return reconstruction_error + kl

        self.vae = Model(self.x, x_star)
        self.vae.compile(optimizer='adam', loss=vae_loss)

    def _set_encoder(self):
        self.encoder = Model(self.x, self.latent_variable)

    def _set_decoder(self):
        z = Input(shape=(self.latent_dim,))
        d1 = self.p_xz(z)
        x_star = self.generator(d1)
        self.decoder = Model(z, x_star)

    def fit(self, x_train, x_test, **kwargs):
        self.initialize()
        self.history = (
            self.vae.fit(x=x_train, y=x_train,
                         batch_size=self.batch_size,
                         validation_data=(x_test, x_test),
                         **kwargs))
        self.finalize()

    def finalize(self):
        self._set_encoder()
        self._set_decoder()

    def encode(self, X):
        z_star = self.encoder.predict(X)
        return z_star

    def decode(self, L):
        reconstruction = self.decoder.predict(L)
        return reconstruction
