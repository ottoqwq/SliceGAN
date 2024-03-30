import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

## Training Utils

def mkdr(proj, proj_dir, Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """
    pth = proj_dir + '/' + proj
    if Training:
        try:
            os.mkdir(pth)
            return pth + '/' + proj
        except FileExistsError:
            print('Directory', pth, 'already exists. Enter new project name or hit enter to overwrite')
            new = input()
            if new == '':
                return pth + '/' + proj
            else:
                pth = mkdr(new, proj_dir, Training)
                return pth
        except FileNotFoundError:
            print('The specified project directory ' + proj_dir + ' does not exist. Please change to a directory that does exist and again')
            sys.exit()
    else:
        return pth + '/' + proj


def weights_init(m):
    """
    Initialises training weights
    :param m: Convolution to be initialized
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)(m.weights[0])
    elif classname.find('BatchNorm') != -1:
        tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)(m.weights[0])
        tf.keras.initializers.Zeros()(m.weights[1])

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, gp_lambda, nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    # sample and reshape random numbers
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], dtype=tf.float32)
    alpha = tf.broadcast_to(alpha, shape=[batch_size, nc, l, l])

    # create interpolate dataset
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = tf.Variable(interpolates, trainable=True)

    # pass interpolates through netD
    with tf.GradientTape() as tape:
        tape.watch(interpolates)
        disc_interpolates = netD(interpolates)
    gradients = tape.gradient(disc_interpolates, interpolates)

    # extract the grads and calculate gp
    gradients = tf.reshape(gradients, [gradients.shape[0], -1])
    gradient_penalty = tf.reduce_mean((tf.norm(gradients, ord=2, axis=1) - 1) ** 2) * gp_lambda
    return gradient_penalty


def calc_eta(steps, time, start, i, epoch, num_epochs):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps:
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch:
    :param num_epochs: total no. of epochs
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))

## Plotting Utils
def post_proc(img, imtype):
    """
    turns one hot image back into grayscale
    :param img: input image
    :param imtype: image type
    :return: plottable image in the same form as the training data
    """
    if imtype == 'colour':
        return np.uint8(255 * (np.swapaxes(img[0], 0, -1)))
    if imtype == 'grayscale':
        return np.uint8(255 * img[0, ..., 0])
    else:
        nphase = img.shape[1]
        return np.uint8(255 * np.argmax(img, axis=1) / (nphase - 1))

def test_plotter(img, slcs, imtype, pth):
    """
    creates a fig with 3*slc subplots showing example slices along the three axes
    :param img: raw input image
    :param slcs: number of slices to take in each dir
    :param imtype: image type
    :param pth: where to save plot
    """
    img = post_proc(img, imtype)[0]
    fig, axs = plt.subplots(slcs, 3)
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :, :], vmin=0, vmax=255)
            axs[j, 1].imshow(img[:, j, :, :], vmin=0, vmax=255)
            axs[j, 2].imshow(img[:, :, j, :], vmin=0, vmax=255)
    elif imtype == 'grayscale':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :], cmap='gray')
            axs[j, 1].imshow(img[:, j, :], cmap='gray')
            axs[j, 2].imshow(img[:, :, j], cmap='gray')
    else:
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :])
            axs[j, 1].imshow(img[:, j, :])
            axs[j, 2].imshow(img[:, :, j])
    plt.savefig(pth + '_slices.png')
    plt.close()

def graph_plot(data, labels, pth, name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """
    for datum, lbl in zip(data, labels):
        plt.plot(datum, label=lbl)
    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def test_img(pth, imtype, netG, nz=64, lf=4, periodic=False):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where
