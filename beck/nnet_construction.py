# Global imports
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, LeakyReLU, add, Flatten, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
import numpy as np

# Local imports
import beck.config as config
from beck.loss import softmax_cross_entropy_with_logits

import tensorflow.keras.losses
tensorflow.keras.losses.softmax_cross_entropy_with_logits = softmax_cross_entropy_with_logits

def conv_layer(x, nnet_args):
    x = Conv2D(
        filters = nnet_args['CONV_FILTERS'],
        kernel_size = nnet_args['CONV_KERNEL_SIZE'],
        data_format = 'channels_first',
        padding = 'same',
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST'])
    )(x)

    x = BatchNormalization(axis = 1)(x)
    x = LeakyReLU()(x)

    return x

def residual_layer(x, nnet_args):
    input_block = x

    x = Conv2D(
        filters = nnet_args['RES_FILTERS'],
        kernel_size = nnet_args['RES_KERNEL_SIZE'],
        data_format = 'channels_first',
        padding = 'same',
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST'])
    )(x)

    x = BatchNormalization(axis = 1)(x)
    x = LeakyReLU()(x)

    x = Conv2D(
        filters = nnet_args['RES_FILTERS'],
        kernel_size = nnet_args['RES_KERNEL_SIZE'],
        data_format = 'channels_first',
        padding = 'same',
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST'])
    )(x)

    x = BatchNormalization(axis = 1)(x)
    x = add([input_block, x])
    x = LeakyReLU()(x)

    return x

def policy_head(x, nnet_args):
    x = Conv2D(
        filters = nnet_args['POLICY_HEAD_FILTERS'],
        kernel_size = nnet_args['POLICY_HEAD_KERNEL_SIZE'],
        data_format = 'channels_first',
        padding = 'same',
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST'])
    )(x)

    x = BatchNormalization(axis = 1)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(
        np.product(nnet_args['OUTPUT_DIM']),
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST']),
        name = 'policy_head'
    )(x)

    return x

def value_head(x, nnet_args):
    x = Conv2D(
        filters = nnet_args['VALUE_HEAD_FILTERS'],
        kernel_size = nnet_args['VALUE_HEAD_KERNEL_SIZE'],
        data_format = 'channels_first',
        padding = 'same',
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST'])
    )(x)

    x = BatchNormalization(axis = 1)(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)

    x = Dense(
        nnet_args['VALUE_HEAD_DENSE_NEURONS'],
        use_bias = False,
        activation = 'linear',
        kernel_regularizer = l2(nnet_args['REG_CONST']),
    )(x)

    x = LeakyReLU()(x)

    x = Dense(
        1,
        use_bias = False, 
        activation = 'tanh',
        kernel_regularizer = l2(nnet_args['REG_CONST']),
        name = 'value_head'
    )(x)

    return x

def build_residual_cnn(nnet_args):
    main_input = Input(shape = nnet_args['INPUT_DIM'], name = 'main_input')

    x = conv_layer(main_input, nnet_args)
    for _ in range(nnet_args['NUM_OF_RESIDUAL_LAYERS']):
        x = residual_layer(x, nnet_args)

    value = value_head(x, nnet_args)
    policy = policy_head(x, nnet_args)

    nnet = Model(inputs = [main_input], outputs = [value, policy])        

    nnet.compile(
        loss = {'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
        optimizer = SGD(lr = nnet_args['LEARNING_RATE'], momentum = nnet_args['MOMENTUM']),
        loss_weights = {'value_head': 0.5, 'policy_head': 0.5}
    )

    return nnet