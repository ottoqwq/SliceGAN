import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

def batch(data, type, l, sf):
    """
    Generate a batch of images randomly sampled from a training microstructure
    :param data: data path
    :param type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    Testing = False
    if type in ['png', 'jpg']:
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            phases = tf.unique(img)[0].numpy()
            data = tf.zeros([32 * 900, len(phases), l, l])
            for i in range(32 * 900):
                x = tf.random.uniform([], 1, x_max - l - 1, dtype=tf.int32)
                y = tf.random.uniform([], 1, y_max - l - 1, dtype=tf.int32)
                # create one channel per phase for one hot encoding
                for cnt, phs in enumerate(phases):
                    img1 = tf.zeros([l, l])
                    img1 = tf.where(img[x:x + l, y:y + l] == phs, 1.0, img1)
                    data = tf.tensor_scatter_nd_update(data, [[i, cnt]], [img1])

            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            dataset = tf.data.Dataset.from_tensor_slices(data)
            datasetxyz.append(dataset)

    elif type == 'colour':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            img = img[::sf, ::sf, :]
            ep_sz = 32 * 900
            data = tf.zeros([ep_sz, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = tf.random.uniform([], 0, x_max - l, dtype=tf.int32)
                y = tf.random.uniform([], 0, y_max - l, dtype=tf.int32)
                # create one channel per phase for one hot encoding
                data = tf.tensor_scatter_nd_update(data, [[i, 0]], [img[x:x + l, y:y + l, 0]])
                data = tf.tensor_scatter_nd_update(data, [[i, 1]], [img[x:x + l, y:y + l, 1]])
                data = tf.tensor_scatter_nd_update(data, [[i, 2]], [img[x:x + l, y:y + l, 2]])
            print('converting')
            if Testing:
                datatest = tf.transpose(data, [0, 2, 3, 1])
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            dataset = tf.data.Dataset.from_tensor_slices(data)
            datasetxyz.append(dataset)

    elif type == 'grayscale':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img / tf.reduce_max(img)
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = tf.zeros([32 * 900, 1, l, l])
            for i in range(32 * 900):
                x = tf.random.uniform([], 1, x_max - l - 1, dtype=tf.int32)
                y = tf.random.uniform([], 1, y_max - l - 1, dtype=tf.int32)
                subim = img[x:x + l, y:y + l]
                data = tf.tensor_scatter_nd_update(data, [[i, 0]], [subim])
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            dataset = tf.data.Dataset.from_tensor_slices(data)
            datasetxyz.append(dataset)

    return datasetxyz


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
