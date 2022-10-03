from numpy import zeros, ones
from numpy.random import randn, randint
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Reshape
from keras.layers import LeakyReLU
from keras.layers import Activation
import pandas as pd
from matplotlib import pyplot
import numpy as np
from math import sqrt
import pickle


opt= tf.optimizers.Adam(1e-4)
def load_aug_data():
    picklein = open( "count_gen_X.pickle", "rb")
    X = pickle.load(picklein)
    picklein = open( "count_gen_Y.pickle", "rb")
    Y = pickle.load(picklein)
    X = np.asarray(X).astype("float32")
    global X_shape
    X_shape=X.shape[1]
    return  X, Y

def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    return z_input

def generate_real_samples(X_train, n_samples):
    ix = randint(0, X_train.shape[0], n_samples)
    X = X_train[ix]
    y = ones((n_samples, 1))
    return X, y

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input = generate_latent_points(latent_dim, n_samples)
    data = generator.predict(z_input)
    y = zeros((n_samples, 1))
    return data, y


def summarize_performance(step, g_model, latent_dim, n_samples=100):
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    X= X.reshape(X.shape[0], X.shape[1])
    DF = pd.DataFrame(X)
    DF.to_csv("data"+ str(step+1)+".csv")

    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s' % (filename2))

def save_plot(examples, n_examples):
    n_dim=int(sqrt(n_examples))
    for i in range(n_examples):
        pyplot.subplot( n_dim,  n_dim, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i, :, ], cmap='gray_r')
    pyplot.show()

def define_discriminator():
    model=load_model("ResNetbest0.hdf5")
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim):
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    in_lat = Input(shape=(latent_dim,))
    gen = Dense(256, kernel_initializer=init)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(512, kernel_initializer=init)(gen)
    gen = Dense(X_shape * 1, kernel_initializer=init)(gen)
    gen = Activation('tanh')(gen)
    out_layer = Reshape((X_shape, 1))(gen)
    model = Model(in_lat, out_layer)
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = tf.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def train(g_model, d_model, gan_model, X_train, latent_dim, n_epochs=100, n_batch=64):
    bat_per_epo = int(X_train.shape[0] / n_batch)
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        X_real, y_real = generate_real_samples(X_train, n_batch)
        d_loss_r, d_acc_r = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_batch)
        d_loss_f, d_acc_f = d_model.train_on_batch(X_fake, y_fake)
        z_input = generate_latent_points(latent_dim, n_batch)
        y_gan = ones((n_batch, 1))
        g_loss, g_acc = gan_model.train_on_batch(z_input, y_gan)
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_loss_r,d_acc_r, d_loss_f,d_acc_f, g_loss,g_acc))
        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, latent_dim)

X_train, Y_train= load_aug_data()
discriminator = define_discriminator()
generator = define_generator(100)
gan_model = define_gan(generator, discriminator)
latent_dim = 100
train(generator, discriminator, gan_model, X_train, latent_dim, n_epochs=1000, n_batch=32)


