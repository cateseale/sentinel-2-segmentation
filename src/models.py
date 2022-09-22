import tensorflow as tf
import tensorflow.keras.backend as K


# U-Net
def unet(activation='relu', init='he_uniform'):

    inputs = tf.keras.layers.Input(shape=(256, 256, 12))
    conv1 = tf.keras.layers.Convolution2D(16, (3, 3), activation=activation, padding='same', kernel_initializer=init)(inputs)
    conv1 = tf.keras.layers.Convolution2D(16, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Convolution2D(32, (3, 3), activation=activation, padding='same', kernel_initializer=init)(pool1)
    conv2 = tf.keras.layers.Convolution2D(32, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Convolution2D(64, (3, 3), activation=activation, padding='same', kernel_initializer=init)(pool2)
    conv3 = tf.keras.layers.Convolution2D(64, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Convolution2D(128, (3, 3), activation=activation, padding='same', kernel_initializer=init)(pool3)
    conv4 = tf.keras.layers.Convolution2D(128, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Convolution2D(256, (3, 3), activation=activation, padding='same',kernel_initializer=init)(pool4)
    conv5 = tf.keras.layers.Convolution2D(256, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.Convolution2D(128, (3, 3), activation=activation, padding='same', kernel_initializer=init)(up6)
    conv6 = tf.keras.layers.Convolution2D(128, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.Convolution2D(64, (3, 3), activation=activation, padding='same',kernel_initializer=init)(up7)
    conv7 = tf.keras.layers.Convolution2D(64, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.Convolution2D(32, (3, 3), activation=activation, padding='same', kernel_initializer=init)(up8)
    conv8 = tf.keras.layers.Convolution2D(32, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv8)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = tf.keras.layers.Convolution2D(16, (3, 3), activation=activation, padding='same', kernel_initializer=init)(up9)
    conv9 = tf.keras.layers.Convolution2D(16, (3, 3), activation=activation, padding='same', kernel_initializer=init)(conv9)

    conv10 = tf.keras.layers.Convolution2D(2, (1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    return model


def unet_with_dropout(dropout_rate=0.5):
    # define the Unet architecture
    inputs = tf.keras.layers.Input(shape=(256, 256, 12))
    conv1 = tf.keras.layers.Convolution2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(inputs)
    conv1 = tf.keras.layers.Convolution2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(dropout_rate)(pool1)

    conv2 = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool1)
    conv2 = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(dropout_rate)(pool2)

    conv3 = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool2)
    conv3 = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(dropout_rate)(pool3)

    conv4 = tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(pool3)
    conv4 = tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(dropout_rate)(pool4)

    conv5 = tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding='same',kernel_initializer='he_uniform')(pool4)
    conv5 = tf.keras.layers.Convolution2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    up6 = tf.keras.layers.Dropout(dropout_rate)(up6)
    conv6 = tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up6)
    conv6 = tf.keras.layers.Convolution2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    up7 = tf.keras.layers.Dropout(dropout_rate)(up7)
    conv7 = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding='same',kernel_initializer='he_uniform')(up7)
    conv7 = tf.keras.layers.Convolution2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    up8 = tf.keras.layers.Dropout(dropout_rate)(up8)
    conv8 = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up8)
    conv8 = tf.keras.layers.Convolution2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv8)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    up9 = tf.keras.layers.Dropout(dropout_rate)(up9)
    conv9 = tf.keras.layers.Convolution2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(up9)
    conv9 = tf.keras.layers.Convolution2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform')(conv9)

    conv10 = tf.keras.layers.Convolution2D(2, (1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    return model


# Unet with lots of batch normalisation
def apply_double_conv(inputs, new_size):
    return _single_conv(_single_conv(inputs, new_size), new_size)


def _single_conv(inputs, new_size):
    outputs = tf.keras.layers.Convolution2D(new_size, (3, 3), padding='same', kernel_initializer='he_uniform')(inputs)
    outputs = tf.keras.layers.BatchNormalization(axis=-1)(outputs)
    return tf.keras.activations.elu(outputs)


def lobnet():
    inputs = tf.keras.layers.Input((256, 256, 12))
    conv1 = apply_double_conv(inputs, 32)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = apply_double_conv(pool1, 64)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = apply_double_conv(pool2, 128)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = apply_double_conv(pool3, 256)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = apply_double_conv(pool4, 512)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = apply_double_conv(up6, 256)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = apply_double_conv(up7, 128)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = apply_double_conv(up8, 64)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = _single_conv(up9, 32)
    conv9 = tf.keras.layers.Convolution2D(32, (3, 3), padding='same', kernel_initializer='he_uniform')(conv9)
    conv9 = tf.keras.layers.BatchNormalization(axis=-1)(conv9)
    conv9 = tf.keras.activations.elu(conv9)
    conv10 = tf.keras.layers.Convolution2D(2, (1, 1), activation='softmax')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model


# Hyperopt
def conv_layer(prev_layer, no_filters, hyperspace, force_ksize=None, STARTING_L2_REG=0.0007):
    """
    Basic convolution layer, parametrized by the hyperspace.

    """
    if force_ksize is not None:
        k = force_ksize
    else:
        k = int(round(hyperspace['conv_kernel_size']))
    return tf.keras.layers.Conv2D(filters=no_filters,
                                  kernel_size=(k, k),
                                  strides=(1, 1),
                                  padding='same',
                                  activation=hyperspace['activation'],
                                  kernel_regularizer=tf.keras.regularizers.l2(STARTING_L2_REG * hyperspace['l2_weight_reg_mult'])
                                  )(prev_layer)


def batch_norm(prev_layer):
    """
    Perform batch normalisation.

    """
    return tf.keras.layers.BatchNormalization()(prev_layer)


def convolution_pooling(prev_layer, no_filters, hyperspace, STARTING_L2_REG=0.0007):
    """
    Pooling with a convolution of stride 2.
    See: https://arxiv.org/pdf/1412.6806.pdf
    """
    current_layer = tf.keras.layers.Conv2D(filters=no_filters,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           activation='linear',
                                           kernel_regularizer=tf.keras.regularizers.l2(STARTING_L2_REG * hyperspace['l2_weight_reg_mult'])
                                           )(prev_layer)

    if hyperspace['use_BN']:
        current_layer = batch_norm(current_layer)

    return current_layer


def inception_reduction(prev_layer, no_filters, hyperspace):
    """
    Reduction block, vaguely inspired from inception.
    See: https://arxiv.org/pdf/1602.07261.pdf
    """
    n_filters_a = int(no_filters * 0.33 + 1)
    n_filters = int(no_filters * 0.4 + 1)

    conv1 = conv_layer(prev_layer, n_filters_a, hyperspace, force_ksize=3)
    conv1 = convolution_pooling(conv1, n_filters, hyperspace)

    conv2 = conv_layer(prev_layer, n_filters_a, hyperspace, 1)
    conv2 = conv_layer(conv2, n_filters, hyperspace, 3)
    conv2 = convolution_pooling(conv2, n_filters, hyperspace)

    conv3 = conv_layer(prev_layer, n_filters, hyperspace, force_ksize=1)
    conv3 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv3)

    current_layer = tf.keras.layers.Concatenate([conv1, conv2, conv3], axis=-1)

    return current_layer


def pooling_layer(prev_layer, no_filters, hyperspace):

    if hyperspace['pooling_type'] == 'all_conv':
        current_layer = convolution_pooling(prev_layer, no_filters, hyperspace)

    elif hyperspace['pooling_type'] == 'inception':
        current_layer = inception_reduction(prev_layer, no_filters, hyperspace)

    elif hyperspace['pooling_type'] == 'avg':
        current_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(prev_layer)

    else:  # 'max'
        current_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(prev_layer)

    return current_layer


def conv_block(prev_layer, no_filters, hyperspace):

    conv_a = tf.keras.layers.Convolution2D(filters=no_filters,
                                               kernel_size=(3, 3),
                                               activation=hyperspace['activation'],
                                               padding='same',
                                               kernel_initializer=hyperspace['initializer'])(prev_layer)
    conv_b = tf.keras.layers.Convolution2D(filters=no_filters,
                                               kernel_size=(3, 3),
                                               activation=hyperspace['activation'],
                                               padding='same',
                                               kernel_initializer=hyperspace['initializer'])(conv_a)
    # pool_a = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_b)
    pool_a = pooling_layer(conv_b, no_filters, hyperspace)

    if hyperspace['add_dropout_layer']:
        pool_a = tf.keras.layers.Dropout(hyperspace['dropout_rate'])(pool_a)

    return pool_a, conv_b


def up_block(prev_layer, skip_connection, no_filters, hyperspace):

    if hyperspace['upsample_mode'] == "TRANSPOSECONV":
        next_layer = tf.keras.layers.concatenate(
            [tf.keras.layers.Conv2DTranspose(int(no_filters), (2, 2), strides=(2, 2), padding='same')(prev_layer),
             skip_connection], axis=3)

        next_layer = conv_layer(next_layer, int(no_filters / 2), hyperspace, force_ksize=3)
        next_layer = conv_layer(next_layer, int(no_filters / 2), hyperspace, force_ksize=3)

    elif hyperspace['upsample_mode'] == "UPSAMPLE2D":
        next_layer = tf.keras.layers.concatenate(
            [tf.keras.layers.UpSampling2D((2, 2))(prev_layer), skip_connection], axis=3)

        next_layer = conv_layer(next_layer, int(no_filters / 2), hyperspace, force_ksize=3)
        next_layer = conv_layer(next_layer, int(no_filters / 2), hyperspace, force_ksize=3)

    return next_layer


def hyperopt_model():

    hyperspace = {
        'batch_size': 32,
        'upsample_mode': "UPSAMPLE2D",
        'optimizer': 'RMSprop',
        'initializer': 'glorot_normal',
        'activation': 'relu',
        'use_BN': False,
        'starting_filter_size': 16,
        'add_dropout_layer': True,
        'dropout_rate': 0.517620637669919,
        'pooling_type': 'all_conv',
        'l2_weight_reg_mult': 0.8427785479730224,
        'conv_kernel_size': 2,
        'loss': 'SDC'
    }

    inputs = tf.keras.layers.Input(shape=(256, 256, 12))

    no_filters = hyperspace['starting_filter_size']

    pool_1, conv_1 = conv_block(inputs, no_filters, hyperspace)
    pool_2, conv_2 = conv_block(pool_1, no_filters * 2, hyperspace)
    pool_3, conv_3 = conv_block(pool_2, no_filters * 4, hyperspace)
    pool_4, conv_4 = conv_block(pool_3, no_filters * 8, hyperspace)

    conv5 = conv_layer(pool_4, no_filters * 16, hyperspace)
    conv5 = conv_layer(conv5, no_filters * 16, hyperspace)

    conv_6 = up_block(conv5, conv_4, no_filters * 16, hyperspace)
    conv_7 = up_block(conv_6, conv_3, no_filters * 8, hyperspace)
    conv_8 = up_block(conv_7, conv_2, no_filters * 4, hyperspace)
    conv_9 = up_block(conv_8, conv_1, no_filters * 2, hyperspace)

    outputs = tf.keras.layers.Convolution2D(2, (1, 1), activation='softmax')(conv_9)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model


# DeepLabv3
def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """
    SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes

    Inputs:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & poinwise convs
        epsilon: epsilon to use in BN layer

    Returns:
          x: output tensor
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise')(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """
    Implements right 'same' padding for even kernel sizes. Without this there is a 1 pixel drift when stride = 2

    Inputs:
        x: input tensor
        filters: num of filters in pointwise convolution
        prefix: prefix before name
        stride: stride at depthwise conv
        kernel_size: kernel size for depthwise convolution
        rate: atrous rate for depthwise convolution
    """
    if stride == 1:

        return tf.keras.layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)

        return tf.keras.layers.Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False, return_skip=False):
    """
    Basic building block of modified Xception network

    Inputs:
        inputs: input tensor
        depth_list: number of filters in each SepConv layer. len(depth_list) == 3
        prefix: prefix before name
        skip_connection_type: one of {'conv','sum','none'}
        stride: stride at last depthwise conv
        rate: atrous rate for depthwise convolution
        depth_activation: flag to use activation between depthwise & pointwise convs
        return_skip: flag to return additional tensor after 2 SepConvs for decoder

        """

    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = tf.keras.layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = tf.keras.layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = tf.keras.layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def _make_divisible(v, divisor, min_value=None):

    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):

    in_channels = inputs.shape[-1]#.value  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand

        x = tf.keras.layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = tf.keras.layers.Activation(tf.nn.relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = tf.keras.layers.Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return tf.keras.layers.Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def Deeplabv3(weights=None, input_tensor=None, input_shape=(256, 256, 12), classes=2, backbone='mobilenetv2', OS=16,
              alpha=1., activation=None):
    """

    Instantiates the Deeplabv3+ architecture
    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.

    Inputs:
        weights: one of 'pascal_voc' (pre-trained on pascal voc), 'cityscapes' (pre-trained on cityscape) or None
            (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`) to use as image input for the model.
        input_shape: shape of input image. format HxWxC PASCAL VOC model was trained on (512,512,3) images. None is
            allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network. One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}. Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
    Returns:
        model: A Keras model instance.

    Raises:
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """

    if not (weights in {'pascal_voc', 'cityscapes', None}):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `pascal_voc`, or `cityscapes` '
                         '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`  or `mobilenetv2` ')

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        img_input = input_tensor

    if backbone == 'xception':
        if OS == 8:
            entry_block3_stride = 1
            middle_block_rate = 2  # ! Not mentioned in paper, but required
            exit_block_rates = (2, 4)
            atrous_rates = (12, 24, 36)
        else:
            entry_block3_stride = 2
            middle_block_rate = 1
            exit_block_rates = (1, 2)
            atrous_rates = (6, 12, 18)

        x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2),
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
        x = tf.keras.layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

        x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
        x = tf.keras.layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu)(x)

        x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)
        x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                   skip_connection_type='conv', stride=2,
                                   depth_activation=False, return_skip=True)

        x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)
        for i in range(16):
            x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                depth_activation=False)

        x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                            depth_activation=False)
        x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                            skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                            depth_activation=True)

    else:
        OS = 8
        first_block_filters = _make_divisible(32 * alpha, 8)
        x = tf.keras.layers.Conv2D(first_block_filters,
                   kernel_size=3,
                   strides=(2, 2), padding='same',
                   use_bias=False, name='Conv')(img_input)
        x = tf.keras.layers.BatchNormalization(
            epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
        x = tf.keras.layers.Activation(tf.nn.relu6, name='Conv_Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=0, skip_connection=False)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, skip_connection=False)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2, skip_connection=True)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, skip_connection=False)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4, skip_connection=True)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5, skip_connection=True)

        # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                expansion=6, block_id=6, skip_connection=False)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=7, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=8, skip_connection=True)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=9, skip_connection=True)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=10, skip_connection=False)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=11, skip_connection=True)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                expansion=6, block_id=12, skip_connection=True)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                expansion=6, block_id=13, skip_connection=False)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=14, skip_connection=True)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=15, skip_connection=True)

        x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    shape_before = tf.shape(x)
    b4 = tf.keras.layers.GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = tf.keras.layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = tf.keras.layers.Activation(tf.nn.relu)(b4)

    # upsample. have to use compat because of the option align_corners
    size_before = tf.keras.backend.int_shape(x)
    b4 = tf.keras.layers.Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                    method='bilinear', align_corners=True))(b4)
    # simple 1x1
    b0 = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = tf.keras.layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = tf.keras.layers.Activation(tf.nn.relu, name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1',
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2',
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3',
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

        # concatenate ASPP branches & project
        x = tf.keras.layers.Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = tf.keras.layers.Concatenate()([b4, b0])

    x = tf.keras.layers.Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = tf.keras.layers.BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
    x = tf.keras.layers.Activation(tf.nn.relu)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        skip_size = tf.keras.backend.int_shape(skip1)
        x = tf.keras.layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        skip_size[1:3],
                                                        method='bilinear', align_corners=True))(x)

        dec_skip1 = tf.keras.layers.Conv2D(48, (1, 1), padding='same',
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = tf.keras.layers.BatchNormalization(
            name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = tf.keras.layers.Activation(tf.nn.relu)(dec_skip1)
        x = tf.keras.layers.Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0',
                       depth_activation=True, epsilon=1e-5)
        x = SepConv_BN(x, 256, 'decoder_conv1',
                       depth_activation=True, epsilon=1e-5)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
        last_layer_name = 'logits_semantic'
    else:
        last_layer_name = 'custom_logits_semantic'

    x = tf.keras.layers.Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
    size_before3 = tf.keras.backend.int_shape(img_input)
    x = tf.keras.layers.Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                    size_before3[1:3],
                                                    method='bilinear', align_corners=True))(x)

    # # Ensure that the model takes into account
    # # any potential predecessors of `input_tensor`.
    # if input_tensor is not None:
    #     inputs = get_source_inputs(input_tensor)
    #
    # else:
    #     inputs = img_input

    if activation in {'softmax', 'sigmoid'}:
        x = tf.keras.layers.Activation(activation)(x)

    model = tf.keras.Model(img_input, x, name='deeplabv3plus')

    # # load weights
    #
    # if weights == 'pascal_voc':
    #     if backbone == 'xception':
    #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH_X,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
    #                                 WEIGHTS_PATH_MOBILE,
    #                                 cache_subdir='models')
    #     model.load_weights(weights_path, by_name=True)
    # elif weights == 'cityscapes':
    #     if backbone == 'xception':
    #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
    #                                 WEIGHTS_PATH_X_CS,
    #                                 cache_subdir='models')
    #     else:
    #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
    #                                 WEIGHTS_PATH_MOBILE_CS,
    #                                 cache_subdir='models')
    #     model.load_weights(weights_path, by_name=True)
    return model
