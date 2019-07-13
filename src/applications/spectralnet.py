'''
spectralnet.py: contains run function for spectralnet
'''
import sys, os, pickle
import tensorflow as tf
import numpy as np
import traceback
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import normalized_mutual_info_score as nmi

import keras.backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda
from keras.optimizers import RMSprop
from sklearn import manifold
from core import train
from core import costs
from core import networks
from core.layer import stack_layers
from core.util import get_scale, print_accuracy, get_cluster_sols, LearningHandler, make_layer_list, train_gen, get_y_preds
import scipy.io as scio
import matplotlib.pyplot as plt
def run_net(data, params):
    #
    # UNPACK DATA
    #

    x_train, y_train, x_val, y_val, x_test, y_test = data['spectral']['train_and_test']
    x_train_unlabeled, y_train_unlabeled, x_train_labeled, y_train_labeled = data['spectral']['train_unlabeled_and_labeled']
    x_val_unlabeled, y_val_unlabeled, x_val_labeled, y_val_labeled = data['spectral']['val_unlabeled_and_labeled']

    if 'siamese' in params['affinity']:
        pairs_train, dist_train, pairs_val, dist_val = data['siamese']['train_and_test']

    # x = np.concatenate((x_train, x_val), axis=0)
    # y = np.concatenate((y_train, y_val), axis=0)

    if len(x_train_labeled):
        y_train_labeled_onehot = OneHotEncoder().fit_transform(y_train_labeled.reshape(-1, 1)).toarray()
    else:
        y_train_labeled_onehot = np.empty((0, len(np.unique(y_val))))

    #
    # SET UP INPUTS
    #

    # create true y placeholder (not used in unsupervised training)
    y_true = tf.placeholder(tf.float32, shape=(None, params['n_clusters']), name='y_true')

    batch_sizes = {
            'Unlabeled': params['batch_size'],
            'Labeled': params['batch_size'],
            'Orthonorm': params.get('batch_size_orthonorm', params['batch_size']),
            }

    input_shape = x_val.shape[1:]

    # spectralnet has three inputs -- they are defined here
    inputs = {
            'Unlabeled': Input(shape=input_shape,name='UnlabeledInput'),
            'Labeled': Input(shape=input_shape,name='LabeledInput'),
            'Orthonorm': Input(shape=input_shape,name='OrthonormInput'),
            }

    #
    # DEFINE AND TRAIN SIAMESE NET
    #

    # run only if we are using a siamese network
    if params['affinity'] == 'siamese':
        siamese_net = networks.SiameseNet(inputs, params['arch'], params.get('siam_reg'), y_true)

        history = siamese_net.train(pairs_train, dist_train, pairs_val, dist_val,
                params['siam_lr'], params['siam_drop'], params['siam_patience'],
                params['siam_ne'], params['siam_batch_size'])

    else:
        siamese_net = None

    #
    # DEFINE AND TRAIN SPECTRALNET
    #

    spectral_net = networks.SpectralNet(inputs, params['arch'],
            params.get('spec_reg'), y_true, y_train_labeled_onehot,
            params['n_clusters'], params['affinity'], params['scale_nbr'],
            params['n_nbrs'], batch_sizes, x_train,siamese_net, x_val, len(x_val_labeled))

    spectral_net.train(
            x_val, x_train_labeled, x_val_unlabeled,x_train,
            params['spec_lr'], params['spec_drop'], params['spec_patience'],
            params['spec_ne'])
    # spectral_net.save
    print("finished training")

    #
    # EVALUATE
    #
    # spectral_net.save('save.h5')
    # spectral_net.load_weight
    # get final embeddings
    x_spectralnet = spectral_net.predict(x_val)

    # get accuracy and nmi
    kmeans_assignments, km = get_cluster_sols(x_spectralnet, ClusterClass=KMeans, n_clusters=params['n_clusters'], init_args={'n_init':10})
    y_spectralnet, _ = get_y_preds(kmeans_assignments, y_val, params['n_clusters'])

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Z_tsne = tsne.fit_transform(x_spectralnet)
    fig = plt.figure()
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], s=2, c=y_val, cmap=plt.cm.get_cmap("jet", 10))
    plt.colorbar(ticks=range(10))
    plt.show()
    scio.savemat('saveddata.mat', {'x': Z_tsne,'y':y_val})
    print_accuracy(kmeans_assignments, y_val, params['n_clusters'])
    from sklearn.metrics import normalized_mutual_info_score as nmi
    nmi_score = nmi(kmeans_assignments, y_val)
    print('NMI: ' + str(np.round(nmi_score, 4)))

    if params['generalization_metrics']:
        x_spectralnet_train = spectral_net.predict(x_train_unlabeled)
        x_spectralnet_test = spectral_net.predict(x_test)
        km_train = KMeans(n_clusters=params['n_clusters']).fit(x_spectralnet_train)
        from scipy.spatial.distance import cdist
        dist_mat = cdist(x_spectralnet_test, km_train.cluster_centers_)
        closest_cluster = np.argmin(dist_mat, axis=1)
        print_accuracy(closest_cluster, y_test, params['n_clusters'], ' generalization')
        nmi_score = nmi(closest_cluster, y_test)
        print('generalization NMI: ' + str(np.round(nmi_score, 3)))

    return x_spectralnet, y_spectralnet

