from slicegan import preprocessing, util
import tensorflow as tf
import time
import matplotlib

def train(pth, imtype, datatype, real_data, Disc, Gen, nc, l, nz, sf):
    """
    train the generator
    :param pth: path to save all files, imgs and data
    :param imtype: image type e.g nphase, colour or gray
    :param datatype: training data format e.g. png, jpg ect
    :param real_data: path to training data
    :param Disc:
    :param Gen:
    :param nc: channels
    :param l: image size
    :param nz: latent vector size
    :param sf: scale factor for training data
    :return:
    """
    if len(real_data) == 1:
        real_data *= 3
        isotropic = True
    else:
        isotropic = False

    print('Loading Dataset...')
    dataset_xyz = preprocessing.batch(real_data, datatype, l, sf)

    ## Constants for NNs
    matplotlib.use('Agg')
    ngpu = 1
    num_epochs = 100

    # batch sizes
    batch_size = 8
    D_batch_size = 8
    # optimiser params for G and D
    lrg = 0.0001
    lrd = 0.0001
    beta1 = 0.9
    beta2 = 0.99
    Lambda = 10
    critic_iters = 5
    workers = 0
    lz = 4
    ##Dataloaders for each orientation
    device = '/GPU:0' if tf.test.is_gpu_available() else '/CPU:0'
    print(device, " will be used.\n")

    # D trained using different data for x, y and z directions
    dataloaderx = tf.data.Dataset.from_tensor_slices(dataset_xyz[0]).shuffle(buffer_size=len(dataset_xyz[0])).batch(batch_size)
    dataloadery = tf.data.Dataset.from_tensor_slices(dataset_xyz[1]).shuffle(buffer_size=len(dataset_xyz[1])).batch(batch_size)
    dataloaderz = tf.data.Dataset.from_tensor_slices(dataset_xyz[2]).shuffle(buffer_size=len(dataset_xyz[2])).batch(batch_size)

    # Create the Generator network
    netG = Gen()
    optG = tf.keras.optimizers.Adam(learning_rate=lrg, beta_1=beta1, beta_2=beta2)

    # Define 1 Discriminator and optimizer for each plane in each dimension
    netDs = []
    optDs = []
    for i in range(3):
        netD = Disc()
        netDs.append(netD)
        optDs.append(tf.keras.optimizers.Adam(learning_rate=lrd, beta_1=beta1, beta_2=beta2))

    disc_real_log = []
    disc_fake_log = []
    gp_log = []
    Wass_log = []

    print("Starting Training Loop...")
    # For each epoch
    start = time.time()
    for epoch in range(num_epochs):
        # sample data for each direction
        for i, (datax, datay, dataz) in enumerate(zip(dataloaderx, dataloadery, dataloaderz), 1):
            dataset = [datax, datay, dataz]
            ### Initialise
            ### Discriminator
            ## Generate fake image batch with G
            noise = tf.random.normal([D_batch_size, nz, lz, lz, lz])
            fake_data = netG(noise)
            # for each dim (d1, d2 and d3 are used as permutations to make 3D volume into a batch of 2D images)
            for dim, (netD, optimizer, data, d1, d2, d3) in enumerate(
                    zip(netDs, optDs, dataset, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                if isotropic:
                    netD = netDs[0]
                    optimizer = optDs[0]
                with tf.GradientTape() as disc_tape:
                    ##train on real images
                    real_data = data
                    out_real = tf.reduce_mean(netD(real_data))
                    ## train on fake images
                    # perform permutation + reshape to turn volume into batch of 2D images to pass to D
                    fake_data_perm = tf.transpose(fake_data, perm=[0, d1, 1, d2, d3])
                    fake_data_perm = tf.reshape(fake_data_perm, [l * D_batch_size, nc, l, l])
                    out_fake = tf.reduce_mean(netD(fake_data_perm))
                    gradient_penalty = util.calc_gradient_penalty(netD, real_data, fake_data_perm[:batch_size],
                                                                          batch_size, l,
                                                                          Lambda, nc)
                    disc_cost = out_fake - out_real + gradient_penalty
                gradients_of_discriminator = disc_tape.gradient(disc_cost, netD.trainable_variables)
                optimizer.apply_gradients(zip(gradients_of_discriminator, netD.trainable_variables))
            #logs for plotting
            disc_real_log.append(out_real.numpy())
            disc_fake_log.append(out_fake.numpy())
            Wass_log.append(out_real.numpy() - out_fake.numpy())
            gp_log.append(gradient_penalty.numpy())
            ### Generator Training
            if i % int(critic_iters) == 0:
                with tf.GradientTape() as gen_tape:
                    errG = 0
                    noise = tf.random.normal([batch_size, nz, lz, lz, lz])
                    fake = netG(noise)

                    for dim, (netD, d1, d2, d3) in enumerate(
                            zip(netDs, [2, 3, 4], [3, 2, 2], [4, 4, 3])):
                        if isotropic:
                            #only need one D
                            netD = netDs[0]
                        # permute and reshape to feed to disc
                        fake_data_perm = tf.transpose(fake, perm=[0, d1, 1, d2, d3])
                        fake_data_perm = tf.reshape(fake_data_perm, [l * batch_size, nc, l, l])
                        output = netD(fake_data_perm)
                        errG -= tf.reduce_mean(output)
                    # Calculate gradients for G
                gradients_of_generator = gen_tape.gradient(errG, netG.trainable_variables)
                optG.apply_gradients(zip(gradients_of_generator, netG.trainable_variables))

            # Output training stats & show imgs
            if i % 25 == 0:
                netG.save_weights(pth + '_Gen.h5')
                netD.save_weights(pth + '_Disc.h5')
                noise = tf.random.normal([1, nz, lz, lz, lz])
                img = netG(noise)
                ###Print progress
                ## calc ETA
                steps = len(dataloaderx)
                util.calc_eta(steps, time.time(), start, i, epoch, num_epochs)
                ###save example slices
                util.test_plotter(img, 5, imtype, pth)
                # plotting graphs
                util.graph_plot([disc_real_log, disc_fake_log], ['real', 'perp'], pth, 'LossGraph')
                util.graph_plot([Wass_log], ['Wass Distance'], pth, 'WassGraph')
                util.graph_plot([gp_log], ['Gradient Penalty'], pth, 'GpGraph')
