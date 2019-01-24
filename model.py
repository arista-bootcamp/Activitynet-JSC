import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications.inception_resnet_v2 import InceptionResNetV2


def _initialize_pretrained_model(base_model_layer='conv_7b'):
    base_model = InceptionResNetV2(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(base_model_layer).output)
    return model


def _add_fc_layer(inputs, training):
    """
    # First FC Layer
    model = tf.layers.dense(inputs=inputs, units=4096)
    model = tf.layers.batch_normalization(model, training=training)
    model = tf.nn.relu(model)

    # Second FC Layer
    model = tf.layers.dense(inputs=model, units=4096)
    model = tf.layers.batch_normalization(model, training=training)
    model = tf.nn.relu(model)
    """
    # Third FC Layer
    model = tf.layers.dense(inputs=inputs, units=1000)
    model = tf.layers.batch_normalization(model, training=training)
    model = tf.nn.relu(model)

    return model


def _add_regular_rnn_layers(inputs, params):
    cell = tf.nn.rnn_cell.LSTMCell

    cells_fw = [cell(params['classes_amount']) for _ in range(params['num_layers'])]

    cells_bw = [cell(params['classes_amount']) for _ in range(params['num_layers'])]

    outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cells_fw,
                                                                   cells_bw=cells_bw,
                                                                   inputs=inputs,
                                                                   dtype=tf.float32,
                                                                   scope="rnn_classification")

    fw, bw = tf.split(outputs, num_or_size_splits=2, name='split', axis=2)
    output = tf.divide(tf.add(fw, bw), 2)

    return output


def _unique_tf(vol):
    y, idx, count = tf.unique_with_counts(vol)
    mode = y[tf.argmax(count)]
    return mode


def model_fn(features, mode, params):
    inputs = features['frames_batch']

    if 'labels_batch' in features:
        labels = features['labels_batch']
    else:
        labels = None

    training = mode == tf.estimator.ModeKeys.TRAIN

    print('*******INPUTS.SHAPE Before FC*******', inputs.shape)
    fc_layers = _add_fc_layer(inputs, training)
    print('*******INPUTS.SHAPE After FC*******', fc_layers.shape)

    logits = _add_regular_rnn_layers(fc_layers, params)
    print('*******LOGITS.SHAPE Before FC*******', logits.shape)

    probabilities = tf.nn.softmax(logits, axis=2, name="softmax_tensor")

    if params['predict_mode'] == 'last':
        pass
    elif params['predict_mode'] == 'mode':
        pass

    y_pred = tf.map_fn(_unique_tf, tf.argmax(probabilities, axis=2))

    predictions = {
        "classes": y_pred,
        "probabilities": probabilities
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    precision = tf.metrics.precision_at_thresholds(labels=labels,
                                                   predictions=probabilities,
                                                   thresholds=list(np.linspace(2 / 101, 10 / 101, 5)))

    mean_precision = tf.metrics.mean(precision)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        tf.summary.scalar('mean_precision_train', tf.math.reduce_mean(precision[1]))

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {'precision': precision, 'mean_precision_eval': mean_precision}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
