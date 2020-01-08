from keras.layers import *
from keras.models import Model
from keras import backend as K
import tensorflow as tf

from .layer import stack_layers
from . import costs


class ConvAE:

    def __init__(self,x,params):
        self.x = x
        self.P = tf.eye(tf.shape(self.x)[0])
        h = x



        filters = params['filters']
        latent_dim = params['latent_dim']
        num_classes = params['n_clusters']
        for i in range(1):
            filters *= 2

            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same')(h)

            h = LeakyReLU(0.2)(h)
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same')(h)

            h = LeakyReLU(0.2)(h)

        for i in range(1):
            filters *= 2
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=2,
                    padding='same')(h)
            h = LeakyReLU(0.2)(h)
            h = Conv2D(filters=filters,
                    kernel_size=3,
                    strides=1,
                    padding='same')(h)
            h = LeakyReLU(0.2)(h)


        h_shape = K.int_shape(h)[1:]
        h = Flatten()(h)

        z_mean = Dense(latent_dim)(h) # p(z|x)的均值
        z_log_var = Dense(latent_dim)(h) # p(z|x)的方差

        self.encoder = Model(x, z_mean)

        z = Input(shape=(latent_dim,))
        h = z
        h = Dense(np.prod(h_shape))(h)
        h = Reshape(h_shape)(h)

        for i in range(2):
            h = Conv2DTranspose(filters=filters,
                                kernel_size=3,
                                strides=1,
                                padding='same')(h)
            h = LeakyReLU(0.2)(h)
            h = Conv2DTranspose(filters=filters,
                                kernel_size=3,
                                strides=2,
                                padding='same')(h)
            h = LeakyReLU(0.2)(h)
            filters //= 2

        x_recon = Conv2DTranspose(filters=1,
                                kernel_size=3,
                                activation='sigmoid',
                                padding='same')(h)

        self.decoder = Model(z, x_recon)

        x_recon1 = self.decoder(z_mean)


        W = costs.knn_affinity(z_mean, params['n_nbrs'], scale=2.62, scale_nbr=params['scale_nbr'])
        W = W - self.P

        self.Dy = tf.placeholder(tf.float32, [None, 10], name='Dy')

        # def GCN(m):
        #     return tf.matmul(LI,m)
        z = Input(shape=(latent_dim,))
        y = Dense(1024, activation='relu')(z)
        # y = Lambda(GCN)(y)
        y = Dense(1024, activation='relu')(y)
        # y = Lambda(GCN)(y)
        y = Dense(512, activation='relu')(y)
        # y = Lambda(GCN)(y)
        y = Dense(num_classes, activation='softmax')(y)

        self.classfier = Model(z, y)  # 隐变量分类器

        y = self.classfier(z_mean)


        layers = [
                  {'type': 'Orthonorm', 'name':'orthonorm'}
                  ]

        outputs = stack_layers(y,layers)


        Dy = costs.squared_distance(outputs)


        loss_SPNet = (K.sum(W * Dy))/1024



        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        x_recon = self.decoder(z)

        def shuffling(x):
            idxs = K.arange(0, K.shape(x)[0])
            idxs = K.tf.random_shuffle(idxs)
            return K.gather(x, idxs)

        z_shuffle = Lambda(shuffling)(z)
        z_z_1 = Concatenate()([z, z])
        z_z_2 = Concatenate()([z, z_shuffle])

        z_in = Input(shape=(latent_dim * 2,))
        z1 = z_in
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(latent_dim, activation='relu')(z1)
        z1 = Dense(1, activation='sigmoid')(z1)

        GlobalDiscriminator = Model(z_in, z1)

        z_z_1_scores = GlobalDiscriminator(z_z_1)
        z_z_2_scores = GlobalDiscriminator(z_z_2)
        global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))
        gaussian = Gaussian(num_classes)
        z_prior_mean = gaussian(z)
        self.vae = Model(x, [x_recon1, z_prior_mean,y])

        z_mean = K.expand_dims(z_mean, 1)
        z_log_var = K.expand_dims(z_log_var, 1)
        lamb = 5
        xent_loss = 1 * K.mean((x - x_recon) ** 2, 0)
        xent1_loss = 0.5 * K.mean((x_recon1 - x_recon) ** 2, 0)
        kl_loss = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean) - K.exp(z_log_var))
        kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1),kl_loss), 0)
        cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
        loss_vae = lamb * K.sum(xent_loss) + lamb * K.sum(xent1_loss)+1.5*K.sum(kl_loss)+1*K.sum(cat_loss)+0.1*K.sum(global_info_loss)
        self.loss=(loss_SPNet+loss_vae)
        self.learning_rate = tf.Variable(0., name='spectral_net_learning_rate')
        self.train_step1 = tf.train.AdamOptimizer().minimize(self.loss,var_list=self.vae.weights)
        K.get_session().run(tf.variables_initializer(self.vae.trainable_weights))

    def train_vae(self, x_train_unlabeled,x_dy,batch_size):
        # create handler for early stopping and learning rate scheduling

        losses = self.train_vae_step(
                return_var=[self.loss],
                updates=[self.train_step1]+self.vae.updates,
                x_unlabeled=x_train_unlabeled,
                inputs=self.x,
                x_dy=x_dy,
                batch_sizes=batch_size,
                batches_per_epoch=10)[0]


        return losses

    def train_vae_step(self,return_var, updates, x_unlabeled, inputs,x_dy,
                   batch_sizes,
                   batches_per_epoch=10):

        return_vars_ = np.zeros(shape=(len(return_var)))
        # train batches_per_epoch batches
        for batch_num in range(0, batches_per_epoch):
            feed_dict = {K.learning_phase(): 1}

            # feed corresponding input for each input_type

            batch_ids = np.random.choice(len(x_unlabeled), size=batch_sizes, replace=False)
            feed_dict[inputs] = x_unlabeled[batch_ids]
            feed_dict[self.Dy]=x_dy[batch_ids]


                        # feed_dict[P]=P[batch_ids]

            all_vars = return_var + updates
            return_vars_ += np.asarray(K.get_session().run(all_vars, feed_dict=feed_dict)[:len(return_var)])

        return return_vars_

class Gaussian(Layer):
    """这是个简单的层，只为定义q(z|y)中的均值参数，每个类别配一个均值。
    输出也只是把这些均值输出，为后面计算loss准备，本身没有任何运算。
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z * 0 + K.expand_dims(self.mean, 0)
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])
