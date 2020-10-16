import tensorflow as tf
import core.common as common


def wBiFPNAdd(features, name, epsilon=1e-4):
    weights = tf.get_variable(name=name, dtype=tf.float32, trainable=True,
                              shape=(len(features),), initializer=tf.constant_initializer(value=1/len(features)))
    weights = tf.nn.relu(weights)
    out = tf.reduce_sum([weights[i] * features[i] for i in range(len(features))], axis=0)
    out = out / (tf.reduce_sum(weights) + epsilon)
    return out


def build_wBiFPN(features, id, trainable, upsample_method, last=False):
    with tf.variable_scope('wBiFPN_{}'.format(id)):
        P2_in, P4_in, P8_in, P16_in, P32_in = features
        P32_out_1 = P32_in
        P16_med_1 = common.convolutional(P32_in, filters_shape=(3, 3,  P32_in.shape[-1],  P16_in.shape[-1]),
                                         trainable=trainable, name='P16_med_1_down')
        P16_med_1 = common.upsample(P16_med_1, name='P16_med_1_up', method=upsample_method)
        P16_med_2 = P16_in
        P16_med = wBiFPNAdd([P16_med_1, P16_med_2], 'P16_med')
        P16_out_3 = P16_in
        P16_out_1 = P16_med
        P8_med_1 = common.convolutional(P16_med, filters_shape=(3, 3,  P16_in.shape[-1],  P8_in.shape[-1]),
                                        trainable=trainable, name='P8_med_1_down')
        P8_med_1 = common.upsample(P8_med_1, name='P8_med_1_up', method=upsample_method)
        P8_med_2 = P8_in
        P8_med = wBiFPNAdd([P8_med_1, P8_med_2], 'P8_med')
        P8_out_1 = P8_med
        P8_out_3 = P8_in

        P4_med_1 = common.convolutional(P8_med, filters_shape=(3, 3,  P8_in.shape[-1],  P4_in.shape[-1]),
                                        trainable=trainable, name='P4_med_1_down')
        P4_med_1 = common.upsample(P4_med_1, name='P4_med_1_up', method=upsample_method)
        P4_med_2 = P4_in
        P4_med = wBiFPNAdd([P4_med_1, P4_med_2], 'P4_med')
        P4_out_1 = P4_med
        P4_out_3 = P4_in
        P2_out_1 = common.convolutional(P4_med, filters_shape=(3, 3,  P4_in.shape[-1],  P2_in.shape[-1]),
                                        trainable=trainable, name='P2_out_1_down')
        P2_out_1 = common.upsample(P2_out_1, name='P2_out_1_up', method=upsample_method)
        P2_out_2 = P2_in
        P2_out = wBiFPNAdd([P2_out_1, P2_out_2], 'P2_out')

        P4_out_2 = common.convolutional(P2_out, filters_shape=(3, 3,  P2_in.shape[-1],  P4_in.shape[-1]),
                                        downsample=True, trainable=trainable, name='P4_out_2')
        P4_out = wBiFPNAdd([P4_out_1, P4_out_2, P4_out_3], 'P4_out')

        P8_out_2 = common.convolutional(P4_out, filters_shape=(3, 3,  P4_in.shape[-1],  P8_in.shape[-1]),
                                        downsample=True, trainable=trainable, name='P8_out_2')
        P8_out = wBiFPNAdd([P8_out_1, P8_out_2, P8_out_3], 'P8_out')

        P16_out_2 = common.convolutional(P8_out, filters_shape=(3, 3,  P8_in.shape[-1],  P16_in.shape[-1]),
                                         downsample=True, trainable=trainable, name='P16_out_2')
        P16_out = wBiFPNAdd([P16_out_1, P16_out_2, P16_out_3], 'P16_out')

        P32_out = None
        if not last:
            P32_out_2 = common.convolutional(P16_out, filters_shape=(3, 3,  P16_in.shape[-1],  P32_in.shape[-1]),
                                             downsample=True, trainable=trainable, name='P32_out_2')
            P32_out = wBiFPNAdd([P32_out_1, P32_out_2], 'P32_out')

    return P2_out, P4_out, P8_out, P16_out, P32_out

