

import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import argparse
from collections import defaultdict

from core.data import get_data
from DSCDAN import run_net


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, help='gpu number to use', default='')
parser.add_argument('--dset', type=str, help='gpu number to use', default='mnist')
args = parser.parse_args()

# SELECT GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

params = defaultdict(lambda: None)

# SET GENERAL HYPERPARAMETERS
general_params = {
        'dset': args.dset,                  # dataset: reuters / mnist
        }
params.update(general_params)

# SET DATASET SPECIFIC HYPERPARAMETERS
if args.dset == 'mnist':
    mnist_params = {
        'n_clusters': 10,                   # number of clusters in data
        'n_nbrs': 5,                        # number of nonzero entries (neighbors) to use for graph Laplacian affinity matrix
        'scale_nbr': 2,                     # neighbor used to determine scale of gaussian graph Laplacian; calculated by
        'batch_size': 1024,                 # batch size for spectral net
        'use_approx': False,                # enable / disable approximate nearest neighbors
        'use_all_data': True,               # enable to use all data for training (no test set)
        'latent_dim': 120,
        'img_dim': 28,
        'filters': 16
        }
    params.update(mnist_params)


data = get_data(params)

# RUN EXPERIMENT
run_net(data, params)


