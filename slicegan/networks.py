import tensorflow as tf
import pickle

def slicegan_nets(pth, Training, imtype, dk, ds, df, dp, gk, gs, gf, gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    # save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)

    # Make nets
    class Generator(tf.keras.Model):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = []
            self.bns = []
            for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                self.convs.append(tf.keras.layers.Conv3DTranspose(gf[lay + 1], k, s, padding='same', use_bias=False))
                self.bns.append(tf.keras.layers.BatchNormalization())

        def call(self, x):
            for conv, bn in zip(self.convs[:-1], self.bns[:-1]):
                x = tf.nn.relu(bn(conv(x)))
            # use tanh if colour or grayscale, otherwise softmax for one hot encoded
            if imtype in ['grayscale', 'colour']:
                out = 0.5 * (tf.tanh(self.convs[-1](x)) + 1)
            else:
                out = tf.nn.softmax(self.convs[-1](x), axis=1)
            return out

    class Discriminator(tf.keras.Model):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = []
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(tf.keras.layers.Conv2D(df[lay + 1], k, s, padding='same', use_bias=False))

        def call(self, x):
            for conv in self.convs[:-1]:
                x = tf.nn.relu(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator

def slicegan_rc_nets(pth, Training, imtype, dk, ds, df, dp, gk, gs, gf, gp):
    """
    Define a generator and Discriminator
    :param Training: If training, we save params, if not, we load params from previous.
    This keeps the parameters consistent for older models
    :return:
    """
    # save params
    params = [dk, ds, df, dp, gk, gs, gf, gp]
    # if fresh training, save params
    if Training:
        with open(pth + '_params.data', 'wb') as filehandle:
            # store the data as binary data stream
            pickle.dump(params, filehandle)
    # if loading model, load the associated params file
    else:
        with open(pth + '_params.data', 'rb') as filehandle:
            # read the data as binary data stream
            dk, ds, df, dp, gk, gs, gf, gp = pickle.load(filehandle)

    # Make nets
    class Generator(tf.keras.Model):
        def __init__(self):
            super(Generator, self).__init__()
            self.convs = []
            self.bns = []
            self.rcconv = tf.keras.layers.Conv3D(gf[-1], 3, 1, padding='valid')
            for lay, (k, s, p) in enumerate(zip(gk, gs, gp)):
                self.convs.append(tf.keras.layers.Conv3DTranspose(gf[lay + 1], k, s, padding='same', use_bias=False))
                self.bns.append(tf.keras.layers.BatchNormalization())

        def call(self, x):
            for lay, (conv, bn) in enumerate(zip(self.convs[:-1], self.bns[:-1])):
                x = tf.nn.relu(bn(conv(x)))
            size = (int(x.shape[1] - 1,) * 2, int(x.shape[2] - 1,) * 2, int(x.shape[3] - 1,) * 2)
            x = tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR)
            out = tf.nn.softmax(self.rcconv(x), axis=1)
            return out

    class Discriminator(tf.keras.Model):
        def __init__(self):
            super(Discriminator, self).__init__()
            self.convs = []
            for lay, (k, s, p) in enumerate(zip(dk, ds, dp)):
                self.convs.append(tf.keras.layers.Conv2D(df[lay + 1], k, s, padding='same', use_bias=False))

        def call(self, x):
            for conv in self.convs[:-1]:
                x = tf.nn.relu(conv(x))
            x = self.convs[-1](x)
            return x

    return Discriminator, Generator
