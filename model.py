import tensorflow as tf
import numpy as np


# Class for batch normalization node
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name,
                                            reuse=tf.AUTO_REUSE  # if tensorflow vesrion < 1.4, delete this line
                                            )


# leaky relu function
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)


class DCGAN:
    # Network Parameters
    def __init__(self, sess, batch_size):
        self.learning_rate = 0.0002

        self.sess = sess

        self.batch_size = batch_size

        self.image_shape = [64, 64, 3]
        self.dim_z = 100
        self.dim_W1 = 1024
        self.dim_W2 = 512
        self.dim_W3 = 256
        self.dim_W4 = 128
        self.dim_W5 = 3

        self.G_W1 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W1, self.dim_z], stddev=0.02), name="G_W1")
        self.G_bn1 = batch_norm(name="G_bn1")

        self.G_W2 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W2, self.dim_W1], stddev=0.02), name='G_W2')
        self.G_bn2 = batch_norm(name="G_bn2")

        self.G_W3 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W3, self.dim_W2], stddev=0.02), name='G_W3')
        self.G_bn3 = batch_norm(name="G_bn3")

        self.G_W4 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W4, self.dim_W3], stddev=0.02), name='G_W4')
        self.G_bn4 = batch_norm(name="G_bn4")

        self.G_W5 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W5, self.dim_W4], stddev=0.02), name='G_W5')

        self.D_W1 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W5, self.dim_W4], stddev=0.02), name='D_W1')

        self.D_W2 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W4, self.dim_W3], stddev=0.02), name='D_W2')
        self.D_bn2 = batch_norm(name="D_bn2")

        self.D_W3 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W3, self.dim_W2], stddev=0.02), name='D_W3')
        self.D_bn3 = batch_norm(name="D_bn3")

        self.D_W4 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W2, self.dim_W1], stddev=0.02), name='D_W4')
        self.D_bn4 = batch_norm(name="D_bn4")

        self.D_W5 = tf.Variable(tf.truncated_normal([4, 4, self.dim_W1, 1], stddev=0.02), name='D_W5')

        self.gen_params = [
            self.G_W1,
            self.G_W2,
            self.G_W3,
            self.G_W4,
            self.G_W5
        ]

        self.discrim_params = [
            self.D_W1,
            self.D_W2,
            self.D_W3,
            self.D_W4,
            self.D_W5
        ]

        self._build_model()

    # Build the Network
    def _build_model(self):
        self.Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])

        self.image_real = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape)
        image_gen = self.generate(self.Z)

        d_real = self.discriminate(self.image_real)
        d_gen = self.discriminate(image_gen)

        self.discrim_cost = -tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_gen))

        self.gen_cost = -tf.reduce_mean(tf.log(d_gen))

        self.train_op_discrim = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.discrim_cost,
                                                                                               var_list=self.discrim_params)
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5).minimize(self.gen_cost,
                                                                                           var_list=self.gen_params)

    def generate(self, Z):
        h1 = tf.reshape(Z, [self.batch_size, 1, 1, self.dim_z])
        h1 = tf.nn.conv2d_transpose(h1, self.G_W1, output_shape=[self.batch_size, 4, 4, self.dim_W1],
                                    strides=[1, 4, 4, 1])
        h1 = tf.nn.relu(self.G_bn1(h1))

        h2 = tf.nn.conv2d_transpose(h1, self.G_W2, output_shape=[self.batch_size, 8, 8, self.dim_W2],
                                    strides=[1, 2, 2, 1])
        h2 = tf.nn.relu(self.G_bn2(h2))

        h3 = tf.nn.conv2d_transpose(h2, self.G_W3, output_shape=[self.batch_size, 16, 16, self.dim_W3],
                                    strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(self.G_bn3(h3))

        h4 = tf.nn.conv2d_transpose(h3, self.G_W4, output_shape=[self.batch_size, 32, 32, self.dim_W4],
                                    strides=[1, 2, 2, 1])
        h4 = tf.nn.relu(self.G_bn4(h4))

        h5 = tf.nn.conv2d_transpose(h4, self.G_W5, output_shape=[self.batch_size, 64, 64, self.dim_W5],
                                    strides=[1, 2, 2, 1])

        x = tf.nn.tanh(h5)
        return x

    def discriminate(self, image):
        h1 = lrelu(tf.nn.conv2d(image, self.D_W1, strides=[1, 2, 2, 1], padding='SAME'))
        h2 = lrelu(self.D_bn2(tf.nn.conv2d(h1, self.D_W2, strides=[1, 2, 2, 1], padding='SAME')))
        h3 = lrelu(self.D_bn3(tf.nn.conv2d(h2, self.D_W3, strides=[1, 2, 2, 1], padding='SAME')))
        h4 = lrelu(self.D_bn4(tf.nn.conv2d(h3, self.D_W4, strides=[1, 2, 2, 1], padding='SAME')))
        h5 = lrelu(tf.nn.conv2d(h4, self.D_W5, strides=[1, 4, 4, 1], padding='SAME'))
        h5 = tf.reshape(h5, [self.batch_size, 1])
        y = tf.nn.sigmoid(h5)
        return y

    # Method for generating the fake images
    def sample_generator(self, noise_z, batch_size=1):
        noise_z = np.array(noise_z).reshape([batch_size, self.dim_z])

        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        h1 = tf.reshape(Z, [batch_size, 1, 1, self.dim_z])
        h1 = tf.nn.conv2d_transpose(h1, self.G_W1, output_shape=[batch_size, 4, 4, self.dim_W1],
                                    strides=[1, 4, 4, 1])
        h1 = tf.nn.relu(self.G_bn1(h1))

        output_shape_l2 = [batch_size, 8, 8, self.dim_W2]
        h2 = tf.nn.conv2d_transpose(h1, self.G_W2, output_shape=output_shape_l2, strides=[1, 2, 2, 1])
        h2 = tf.nn.relu(self.G_bn2(h2))

        output_shape_l3 = [batch_size, 16, 16, self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.G_W3, output_shape=output_shape_l3, strides=[1, 2, 2, 1])
        h3 = tf.nn.relu(self.G_bn3(h3))

        output_shape_l4 = [batch_size, 32, 32, self.dim_W4]
        h4 = tf.nn.conv2d_transpose(h3, self.G_W4, output_shape=output_shape_l4, strides=[1, 2, 2, 1])
        h4 = tf.nn.relu(self.G_bn4(h4))

        output_shape_l5 = [batch_size, 64, 64, self.dim_W5]
        h5 = tf.nn.conv2d_transpose(h4, self.G_W5, output_shape=output_shape_l5, strides=[1, 2, 2, 1])

        x = tf.nn.tanh(h5)

        generated_samples = self.sess.run(x, feed_dict={Z: noise_z})
        generated_samples = (generated_samples + 1.) / 2.
        return generated_samples

    # Train Generator and return the loss
    def train_gen(self, noise_z):
        _, loss_val_G = self.sess.run([self.train_op_gen, self.gen_cost], feed_dict={self.Z: noise_z})
        return loss_val_G

    # Train Discriminator and return the loss
    def train_discrim(self, batch_xs, noise_z):
        _, loss_val_D = self.sess.run([self.train_op_discrim, self.discrim_cost],
                                      feed_dict={self.image_real: batch_xs, self.Z: noise_z})
        return loss_val_D
