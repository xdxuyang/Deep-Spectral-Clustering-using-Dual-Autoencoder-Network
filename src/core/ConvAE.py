import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
import imageio,os
#from keras.datasets import mnist
from keras import losses
from sklearn.cluster.k_means_ import KMeans
from sklearn import manifold
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn import metrics
from keras.datasets import mnist as mnist
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def ConvAE(img_dim,latent_dim,filters,intermediate_dim,num_classes):
    x = Input(shape=(img_dim, img_dim, 1))
    h = x
    y =Input(shape=(img_dim, img_dim, 1))
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

        # feature_map = h # 截断到这里，认为到这里是feature_map（局部特征）
        # feature_map_encoder = Model(x, h)

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

    encoder = Model(x, z_mean)

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

    decoder = Model(z, x_recon)

    x_recon1 = decoder(z_mean)

    z = Input(shape=(latent_dim,))
    y = Dense(intermediate_dim, activation='relu')(z)
    y = Dense(num_classes, activation='softmax')(y)

    classfier = Model(z, y)  # 隐变量分类器
    y = classfier(z_mean)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    x_recon = decoder(z)

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
    vae = Model(x, [x_recon, z_prior_mean, y])
    z_mean = K.expand_dims(z_mean, 1)
    z_log_var = K.expand_dims(z_log_var, 1)

    lamb = 5  # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
    xent_loss = 0.5 * K.mean((x - x_recon1) ** 2, 0)
    xent1_loss = 1 * K.mean((x_recon1 - x_recon) ** 2, 0)
        # celoss= 0.5*K.mean(losses.kullback_leibler_divergence(y,y1))
    kl_loss = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean) - K.exp(z_log_var))
    kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
    cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
    vae_loss = lamb * K.sum(xent_loss) + lamb * K.sum(xent1_loss) + 1 * K.sum(kl_loss) + 1 * K.sum(cat_loss) + 0.01 * K.sum(global_info_loss)

    return vae,vae_loss,encoder,decoder,classfier


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



batch_size = 100
latent_dim = 120
epochs = 100
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256


# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
y_train_ = y_train_.reshape(-1)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))


vae,vae_loss,encoder,decoder,classfier = ConvAE(img_dim,latent_dim,filters,intermediate_dim,num_classes)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()



hist=vae.fit(x=x_train,y=y_train_,
        shuffle=True,
        verbose=1,
        batch_size=batch_size,
        validation_data=(x_test, None))



