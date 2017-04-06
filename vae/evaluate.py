import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class MnistMetrics:
    def __init__(self, vae, x_test, y_test, logit=False):
        L = vae.encoder.predict(x_test, batch_size=16)
        self.L = L
        self.vae = vae
        self.df = pd.DataFrame(L)
        self.df["label"] = y_test
        self.df = self.df.set_index("label").sort_index()

        self.df = pd.concat([self.df.loc[i].mean() for i in range(10)], 1)
        self.df.columns = ["label_" + str(i) for i in self.df.columns]

        if logit:
            self.df = self.df.apply(np.exp)
            self.df = self.df / self.df.sum(0)

    def plot_bar(self):
        self.df.plot(kind="bar", subplots=(10), layout=(5, 2),
                     figsize=(10, 10), legend=False)

    def plot_mean_digits(self):
        for i, e in enumerate(self.df.columns):
            plt.subplot(1, 10, 1 + i)
            dat = np.array([list(self.df[e])])
            img = self.vae.decoder.predict(dat).reshape(28, 28)
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    def plot_transition(self, l1, l2, n=10):
        for i, e in enumerate(np.linspace(0, 1, n)):
            dat = (1 - e) * self.df[l1] + e * self.df[l2]
            dat = np.array([list(dat)])
            plt.subplot(1, n, 1 + i)
            img = self.vae.decoder.predict(dat).reshape(28, 28)
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    def plot_random_transition(self, n=10):
        l1 = random.choice(self.df.columns)
        l2 = random.choice(self.df.columns)
        gs1 = gridspec.GridSpec(n, 2)
        for i, e in enumerate(np.linspace(0, 1, n)):
                dat = (1 - e) * self.df[l1] + e * self.df[l2]
                dat = np.array([list(dat)])
                plt.subplot(gs1[i, 0])
                img = self.vae.decoder.predict(dat).reshape(28, 28)
                fig = plt.imshow(img)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.subplot(gs1[i, 1:])
                fig = plt.plot(range(n), dat[0])
                fig[0].axes.get_xaxis().set_visible(False)
                fig[0].axes.get_yaxis().set_visible(False)

    def plot_random_samples(self, n, m, base=None, sigma=1, logit=True):
        for i in range(m * n):
            dat = np.zeros(self.vae.latent_dim)
            dat += np.random.normal(0, sigma, self.vae.latent_dim)

            if logit:
                dat = np.exp(dat)
                dat /= dat.sum()

            plt.subplot(n, m, 1 + i)
            dat = np.array([dat])
            img = self.vae.decoder.predict(dat).reshape(28, 28)
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)

    def plot_from(self, L):
        dat = np.array([L])
        img = self.vae.decoder.predict(dat).reshape(28, 28)
        fig = plt.imshow(img)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

    def plot_onehots(self):
        n = self.vae.latent_dim
        for i in range(n):
            x = np.zeros(self.vae.latent_dim)
            x[i] = 1
            plt.subplot(1, n, 1 + i)
            dat = np.array([x])

            img = self.vae.decoder.predict(dat).reshape(28, 28)
            fig = plt.imshow(img)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
