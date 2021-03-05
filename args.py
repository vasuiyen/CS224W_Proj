"""
Command-line arguments for train.py, test.py.

"""

import argparse

def get_train_args():
    """
    Get arguments needed in train.py.
    """
    parser = argparse.ArgumentParser('Train a model')

    add_common_args(parser)
    add_train_test_args(parser)

    parser.add_argument('--name',
                        '-n',
                        type=str,
                        choices=('GCN', 'RecurrentGraphNeuralNet'),
                        default="RecurrentGraphNeuralNet",
                        help='Name of the class model. Also used to identify subdir or test run.')   

    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs for which to train.')

    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Learning rate.')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Learning rate.')

    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--metric_name',
                        type=str,
                        default='acc',
                        choices=('loss', 'acc'),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--maximize_metric',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use learn rate scheduler.')    

    parser.add_argument('--eval_steps',
                        type=int,
                        default=5000,
                        help='Number of steps between successive evaluations.')

    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=5,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument('--optimizer',
                        type=str,
                        default='Adam',
                        choices=('Adam', 'Adadelta', 'Adagrad', 'SGD'),
                        help='Name of dev metric to determine best checkpoint.')

    parser.add_argument('--use_lr_scheduler',
                        type=lambda s: s.lower().startswith('t'),
                        default=True,
                        help='Whether to use learn rate scheduler.')                     

    parser.add_argument('--dropout',
                        type=int,
                        default=0.3,
                        help='Maximum number of checkpoints to keep on disk.')

    parser.add_argument('--loss_type',
                        type=str,
                        default='NLL',
                        choices=('L1', 'MSE', 'BCELogit', 'BCE', 'NLL', 'CrossEntropy'),
                        help='Name of dev metric to determine best checkpoint.')
    
    parser.add_argument('--hidden-dim',
                        type=int,
                        default=32,
                        help="Number of hidden dimensions to use in IGNN")
    parser.add_argument('--debug',
                        action = 'store_true',
                        help = "Turn on debugging for the RGNN")

    args = parser.parse_args()

    return args


def get_test_args():
    """
    Get arguments needed in test.py.
    """
    parser = argparse.ArgumentParser('Test a trained model')

    add_common_args(parser)
    add_train_test_args(parser)

    args = parser.parse_args()
    
    return args

def add_common_args(parser):

    """
    Add arguments common to all scripts: train.py, test.py
    """

    parser.add_argument('--dataset',
                        type=str,
                        choices=('ogbn-products', 'ogbn-arxiv', 'pyg-karate', 'pyg-cora'),
                        default='ogbn-arxiv')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./save/',
                        help='Base directory for saving information.')

    parser.add_argument('--num_workers',
                        type=int,
                        default=0,
                        help='Number of sub-processes to use per data loader.')
    

def add_train_test_args(parser):
    """
    Add arguments common to train.py and test.py
    """           

    parser.add_argument('--load_path',
                        type=str,
                        default=None,
                        help='Path to load as a model checkpoint.')

    parser.add_argument('--seed',
                        type=int,
                        default=224,
                        help='Random seed for reproducibility.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=512,
                        help='Batch size per GPU. Scales automatically when \
                                  multiple GPUs are available.')

    parser.add_argument('--data_shuffle',
                        type=bool,
                        default=True,
                        help='Shuffle the data.')

    parser.add_argument('--hidden_dim',
                        type=int,
                        default=32,
                        help='Hidden layers dimension size.')

    parser.add_argument('--num_layers',
                        type=int,
                        default=3,
                        help='Number of layers.')
