import math
import os

import PIL
import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers

from src.learner.VAE import VariationalAutoEncoder

tf.random.set_seed(10)
np.random.seed(10)


class LeakyRelu(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, x):
        return tf.nn.leaky_relu(x, alpha=self.alpha)


class Encoder(tf.keras.Model):
    """ Maps mnist digits to (z_mean, z_log_var, z) """
    def __init__(self, latent_dim, name="encoder", alpha=0.1, **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv0 = layers.Conv2D(8, 3, padding="same", activation="relu")
        self.conv1 = layers.Conv2D(32, 3, strides=2, padding="same", activation=LeakyRelu(alpha))
        self.conv2 = layers.Conv2D(64, 3, strides=2, padding="same", activation=LeakyRelu(alpha))
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(16)
        self.mean = layers.Dense(latent_dim, name="z_mean")
        self.logvar = layers.Dense(latent_dim, name="z_log_var")

    def call(self, inputs, **kwargs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        z_mean = self.mean(x)
        z_log_var = self.logvar(x)
        return z_mean, z_log_var


class Decoder(tf.keras.Model):
    """ Converts z back to readable digit """
    def __init__(self, name="decoder", alpha=0.1, **kwargs):
        super(Decoder, self).__init__(name, **kwargs)
        self.dense1 = layers.Dense(25 * 25 * 64)
        self.reshape = layers.Reshape((25, 25, 64))
        self.convt1 = layers.Conv2DTranspose(64, 3, activation=LeakyRelu(alpha), strides=2, padding="same")
        self.convt2 = layers.Conv2DTranspose(32, 3, activation=LeakyRelu(alpha), strides=2, padding="same")
        self.convt3 = layers.Conv2DTranspose(3, 3, activation="tanh", padding="same")

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        return x


class VAEImages(object):
    def __init__(self, input_dir, obj_names, img_size=(100, 100), batch_size=100, epochs=30,
                 validation_split=0.2, latent_dim=2):
        self.input_dir = input_dir
        self.obj_names = obj_names
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.validation_split = validation_split
        self.vae = None

    def plotImgs(self):
        counts = np.zeros(len(self.obj_names), dtype=np.int32)
        nrow = 2
        fig, axs = plt.subplots(nrow, math.ceil(len(self.obj_names) / nrow))
        for i, obj in enumerate(self.obj_names):
            dname = os.path.join(self.input_dir, obj)
            obj_list = os.listdir(dname)
            counts[i] = len(obj_list)
            rand_img = PIL.Image.open(os.path.join(dname, obj_list[0]))
            col, row = divmod(i, nrow)
            axs[row, col].imshow(np.array(rand_img))
            axs[row, col].set_xticks([])
            axs[row, col].set_yticks([])
            axs[row, col].set_title(obj)
        plt.show()
        df = pd.DataFrame({"Object": self.obj_names, "Count": counts})
        print(df)

    def train_vae(self):
        encoder = Encoder(self.latent_dim)
        decoder = Decoder()
        vae = VariationalAutoEncoder(encoder, decoder, from_logits=True, cross_entropy_loss=False, kl_loss_weight=0.1)
        train_dataset = tf.keras.utils.image_dataset_from_directory(self.input_dir, image_size=self.img_size,
                                                                    batch_size=self.batch_size, seed=10,
                                                                    validation_split=self.validation_split,
                                                                    subset="training")
        for batch_num, train_batch in enumerate(train_dataset):
            img_data = train_batch[0].numpy().astype(np.float32) / 255.0
            loss = vae.fit(img_data, epochs=self.epochs)
            print("Batch %d, final loss: %f" % (batch_num+1, loss))
        self.vae = vae

    def test_vae(self):
        assert self.vae, "VAE needs to be trained first"
        valid_dataset = tf.keras.utils.image_dataset_from_directory(self.input_dir, image_size=self.img_size,
                                                                    batch_size=self.batch_size, seed=10,
                                                                    validation_split=self.validation_split,
                                                                    subset="validation")
        class_names = valid_dataset.class_names
        mean_x, mean_y, mean_z = [], [], []
        lvx, lvy, lvz, label, r_mimg, b_mimg, g_mimg = [], [], [], [], [], [], []
        for batch_num, valid_batch in enumerate(valid_dataset):
            img_data = valid_batch[0].numpy().astype(np.float32) / 255.0
            labels = valid_batch[1]
            mean, log_var = self.vae.encoder(img_data)
            mean = mean.numpy()
            log_var = log_var.numpy()
            mean_x.extend(mean[:, 0])
            mean_y.extend(mean[:, 1])
            mean_z.extend(mean[:, 2])
            lvx.extend(log_var[:, 0])
            lvy.extend(log_var[:, 1])
            lvz.extend(log_var[:, 2])
            label.extend([class_names[l] for l in labels])

        df = pd.DataFrame({"Label": label, "Mean(X)": mean_x, "Mean(Y)": mean_y, "Mean(Z)": mean_z,
                           "LogVar(X)": lvx, "LogVar(Y)": lvy, "LogVar(Z)": lvz,
                           })

        sns.pairplot(data=df[["Mean(X)", "Mean(Y)", "Mean(Z)", "Label"]], hue="Label")
        plt.show()


        sns.pairplot(data=df[["LogVar(X)", "LogVar(Y)", "LogVar(Z)", "Label"]], hue="Label")
        plt.show()


if __name__ == "__main__":
    input_dir = r"C:\prog\cygwin\home\samit_000\RLPy\data\kaggle_images\natural_images"
    objs = ["airplane", "car", "cat", "dog", "flower", "fruit", "motorbike", "person"]
    vae_imgs = VAEImages(input_dir, objs, latent_dim=3)
    vae_imgs.plotImgs()
    vae_imgs.train_vae()
    vae_imgs.test_vae()