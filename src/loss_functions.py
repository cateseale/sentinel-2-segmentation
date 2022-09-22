import tensorflow as tf
import tensorflow.keras.backend as K


# Sorensen Dice Loss
def sorensen_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    y_true_f = tf.cast(y_true_f, tf.float32)

    intersection = K.sum(y_true_f * y_pred_f)
    coef = (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    return coef


def sorensen_dice_coef_loss(y_true, y_pred):
    return 1 - sorensen_dice_coef(y_true, y_pred)


# Sobel Loss
sobel_filter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                          [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                          [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])


def expanded_sobel(input_tensor):
    # if you're using 'channels_first', use inputTensor[0,:,0,0]
    input_channels = K.reshape(K.ones_like(input_tensor[0, 0, 0, :]), (1, 1, -1, 1))
    input_channels = tf.cast(input_channels, tf.float32)

    return sobel_filter * input_channels


def sobel_loss(y_true, y_pred):
    filt = expanded_sobel(y_true)
    filt = tf.cast(filt, tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    sobel_true = K.depthwise_conv2d(y_true, filt)
    sobel_pred = K.depthwise_conv2d(y_pred, filt)

    return K.mean(K.square(sobel_true - sobel_pred))