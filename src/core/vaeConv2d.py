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
from keras.datasets import fashion_mnist as mnist
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 100
latent_dim = 120
epochs = 100
num_classes = 10
img_dim = 28
filters = 16
intermediate_dim = 256


# 加载MNIST数据集
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))

# y_train_ = y_train_.reshape(-1)
# from vaeConv2d import encoder,vae
#
# vae.load_weights('vae_mnist1.h5')
#
# x_m = encoder.predict(x_train)
# 搭建模型
x = Input(shape=(img_dim, img_dim, 1))
h = x

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

encoder = Model(x, z_mean) # 通常认为z_mean就是所需的隐变量编码


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


decoder = Model(z, x_recon) # 解码器
generator = decoder
x_recon1 = decoder(z_mean)

z = Input(shape=(latent_dim,))
y = Dense(intermediate_dim, activation='relu')(z)
y = Dense(num_classes, activation='softmax')(y)

classfier = Model(z, y) # 隐变量分类器
y = classfier(z_mean)

# 重参数技巧
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
x_recon = decoder(z)

# y1 = classfier(z)

def shuffling(x):
    idxs = K.arange(0, K.shape(x)[0])
    idxs = K.tf.random_shuffle(idxs)
    return K.gather(x, idxs)

# 与随机采样的特征拼接（全局）
z_shuffle = Lambda(shuffling)(z)
z_z_1 = Concatenate()([z, z])
z_z_2 = Concatenate()([z, z_shuffle])

# 与随机采样的特征拼接（局部）
# feature_map_shuffle = Lambda(shuffling)(feature_map)
# z_samples_repeat = RepeatVector(14 * 14)(z)
# z_samples_map = Reshape((14, 14, latent_dim))(z_samples_repeat)
# z_f_1 = Concatenate()([z_samples_map, feature_map])
# z_f_2 = Concatenate()([z_samples_map, feature_map_shuffle])

# 全局判别器
z_in = Input(shape=(latent_dim*2,))
z1 = z_in
z1 = Dense(latent_dim, activation='relu')(z1)
z1 = Dense(latent_dim, activation='relu')(z1)
z1 = Dense(latent_dim, activation='relu')(z1)
z1 = Dense(1, activation='sigmoid')(z1)

GlobalDiscriminator = Model(z_in, z1)

z_z_1_scores = GlobalDiscriminator(z_z_1)
z_z_2_scores = GlobalDiscriminator(z_z_2)
global_info_loss = - K.mean(K.log(z_z_1_scores + 1e-6) + K.log(1 - z_z_2_scores + 1e-6))


# 局部判别器
# z_in = Input(shape=(None, None, latent_dim*2))
# z1 = z_in
# z1 = Dense(latent_dim, activation='relu')(z1)
# z1 = Dense(latent_dim, activation='relu')(z1)
# z1 = Dense(latent_dim, activation='relu')(z1)
# z1 = Dense(1, activation='sigmoid')(z1)
#
# LocalDiscriminator = Model(z_in, z1)
#
# z_f_1_scores = LocalDiscriminator(z_f_1)
# z_f_2_scores = LocalDiscriminator(z_f_2)
# local_info_loss = - K.mean(K.log(z_f_1_scores + 1e-6) + K.log(1 - z_f_2_scores + 1e-6))



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

gaussian = Gaussian(num_classes)
z_prior_mean = gaussian(z)


# 建立模型
vae = Model(x, [x_recon, z_prior_mean, y])
# vae.load_weights('vae_mnist_2.h5')
# 下面一大通都是为了定义loss
z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)




lamb = 5 # 这是重构误差的权重，它的相反数就是重构方差，越大意味着方差越小。
xent_loss = 0.5 * K.mean((x - x_recon1)**2, 0)
xent1_loss = 1 * K.mean((x_recon1 - x_recon)**2, 0)
# celoss= 0.5*K.mean(losses.kullback_leibler_divergence(y,y1))
kl_loss = - 0.5 * (1 + z_log_var - K.square(z_mean - z_prior_mean) - K.exp(z_log_var))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y, 1), kl_loss), 0)
cat_loss = K.mean(y * K.log(y + K.epsilon()), 0)
vae_loss = lamb * K.sum(xent_loss)+lamb*K.sum(xent1_loss) + 1*K.sum(kl_loss) + 1*K.sum(cat_loss)+0.01*K.sum(global_info_loss)


vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()


# hist=vae.fit(x_train,
#         shuffle=True,
#         epochs=100,
#         verbose=1,
#         batch_size=batch_size)
# vae.save_weights('vae_famnist1.h5')
# # with open('log_sgd_big_32.txt','w') as f:
# #     f.write(str(hist.history))
#
#
# means = K.eval(gaussian.mean)
# x_train_encoded = encoder.predict(x_train)
# y_train_pred = classfier.predict(x_train_encoded).argmax(axis=1)
# x_test_encoded = encoder.predict(x_test)
# y_test_pred = classfier.predict(x_test_encoded).argmax(axis=1)
#
#
# def cluster_sample(path, category=0):
#     """观察被模型聚为同一类的样本
#     """
#     n = 8
#     figure = np.zeros((img_dim * n, img_dim * n))
#     idxs = np.where(y_train_pred == category)[0]
#     for i in range(n):
#         for j in range(n):
#             digit = x_train[np.random.choice(idxs)]
#             digit = digit.reshape((img_dim, img_dim))
#             figure[i * img_dim: (i + 1) * img_dim,
#             j * img_dim: (j + 1) * img_dim] = digit
#     imageio.imwrite(path, figure * 255)
#
#
# def random_sample(path, category=0, std=1):
#     """按照聚类结果进行条件随机生成
#     """
#     n = 8
#     figure = np.zeros((img_dim * n, img_dim * n))
#     for i in range(n):
#         for j in range(n):
#             noise_shape = (1, latent_dim)
#             z_sample = np.array(np.random.randn(*noise_shape)) * std + means[category]
#             x_recon = generator.predict(z_sample)
#             digit = x_recon[0].reshape((img_dim, img_dim))
#             figure[i * img_dim: (i + 1) * img_dim,
#             j * img_dim: (j + 1) * img_dim] = digit
#     imageio.imwrite(path, figure * 255)
#
#
# if not os.path.exists('samples'):
#     os.mkdir('samples')
#
# for i in range(10):
#     cluster_sample(u'samples/聚类类别_%s.png' % i, i)
#     random_sample(u'samples/类别采样_%s.png' % i, i)
#
# print(min(y_train_))
# print(min(y_train_pred))
# right = 0.
# for i in range(10):
#     _ = np.bincount(y_train_[y_train_pred == i])
#     right += _.max()
#
# print ('train acc: %s' % (right / len(y_train_)))
#
#
# right = 0.
# for i in range(10):
#     _ = np.bincount(y_test_[y_test_pred == i])
#     right += _.max()
#
# print('test acc: %s' % (right / len(y_test_)))
#
#
# def evaluateKMeans1(data, labels, nclusters):
#     '''
#     Clusters data with kmeans algorithm and then returns the string containing method name and metrics, and also the evaluated cluster centers
#     :param data: Points that need to be clustered as a numpy array
#     :param labels: True labels for the given points
#     :param nclusters: Total number of clusters
#     :param method_name: Name of the method from which the clustering space originates (only used for printing)
#     :return: Formatted string containing metrics and method name, cluster centers
#     '''
#     kmeans = KMeans(n_clusters=nclusters, n_init=50)
#     kmeans.fit(data)
#     return getClusterMetricString(labels, kmeans.labels_), kmeans.cluster_centers_
#
#
# def getClusterMetricString( labels_true, labels_pred):
#     '''
#     Creates a formatted string containing the method name and acc, nmi metrics - can be used for printing
#     :param method_name: Name of the clustering method (just for printing)
#     :param labels_true: True label for each sample
#     :param labels_pred: Predicted label for each sample
#     :return: Formatted string containing metrics and method name
#     '''
#     acc = cluster_acc(labels_true, labels_pred)
#     nmi = metrics.normalized_mutual_info_score(labels_true, labels_pred)
#     print(acc, nmi)
#     return ' %8.3f     %8.3f' % ( acc, nmi)
# def cluster_acc(y_true, y_pred):
#     '''
#     Uses the hungarian algorithm to find the best permutation mapping and then calculates the accuracy wrt
#     Implementation inpired from https://github.com/piiswrong/dec, since scikit does not implement this metric
#     this mapping and true labels
#     :param y_true: True cluster labels
#     :param y_pred: Predicted cluster labels
#     :return: accuracy score for the clustering
#     '''
#     print(y_true)
#     print(y_pred)
#     D = int(max(y_pred.max(), y_true.max()) + 1)
#     w = np.zeros((D, D), dtype=np.int32)
#     for i in range(y_pred.size):
#         idx1 = int(y_pred[i])
#         idx2 = int(y_true[i])
#         w[idx1, idx2] += 1
#     ind = linear_assignment(w.max() - w)
#     return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
#
# getClusterMetricString(y_train_,y_train_pred)
