import os
import argparse
import tensorflow as tf
import numpy as np
import dbread as db
from model import DCGAN
import scipy.misc

parser = argparse.ArgumentParser(description='Easy Implementation of DCGAN')

# parameters
parser.add_argument('--filelist', type=str, default='filelist.txt')
parser.add_argument('--out_dir', type=str, default='./output')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)


# Function for save the generated result
def save_visualization(X, nh_nw, save_path='./vis/sample.jpg'):
    nh, nw = nh_nw
    h, w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh, w * nw, 3))

    for n, x in enumerate(X):
        j = int(n / nw)
        i = int(n % nw)
        img[j * h:j * h + h, i * w:i * w + w, :] = x

    scipy.misc.imsave(save_path, img)


def main():
    args = parser.parse_args()
    filelist_dir = args.filelist
    output_dir = args.out_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_epoch = args.epochs
    batch_size = args.batch_size
    n_noise = 100

    database = db.DBreader(filelist_dir, batch_size, resize=[64, 64, 3], labeled=False)

    sess = tf.Session()
    model = DCGAN(sess, batch_size)
    sess.run(tf.global_variables_initializer())

    total_batch = database.total_batch

    visualization_num = 14 * 14
    noise_test = np.random.normal(size=(visualization_num, n_noise))

    loss_D = 0.0
    loss_G = 0.0
    for epoch in range(total_epoch):
        for step in range(total_batch):
            batch_xs = database.next_batch()            # Get the next batch
            batch_xs = batch_xs * (2.0 / 255.0) - 1     # normalization
            noise_g = np.random.normal(size=(batch_size, n_noise))
            noise_d = np.random.normal(size=(batch_size, n_noise))

            # Train Generator twice while training Discriminator once for first 200 steps
            if epoch == 0 and step < 200:
                adventage = 2
            else:
                adventage = 1

            if step % adventage == 0:
                loss_D = model.train_discrim(batch_xs, noise_d)     # Train Discriminator and get the loss value
            loss_G = model.train_gen(noise_g)                       # Train Generator and get the loss value

            print('Epoch: [', epoch, '/', total_epoch, '], ', 'Step: [', step, '/', total_batch, '], D_loss: ',
                  loss_D, ', G_loss: ', loss_G)

            if step == 0 or (step + 1) % 10 == 0:
                generated_samples = model.sample_generator(noise_test, batch_size=visualization_num)
                savepath = output_dir + '/output_' + 'EP' + str(epoch).zfill(3) + "_Batch" + str(step).zfill(6) + '.jpg'
                save_visualization(generated_samples, (14, 14), save_path=savepath)


if __name__ == "__main__":
    main()
