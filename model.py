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


def _unique_tf(vol, index=False):
    y, idx, count = tf.unique_with_counts(vol)
    count_max = tf.argmax(count)
    mode = y[count_max]
    if index:
        index_frame = tf.argmax(tf.cumsum(tf.cast(tf.equal(mode, vol), tf.int32)))
        return index_frame
    return mode


def _get_frame(vol, frame=None):
    vol = tf.unstack(vol, axis=0)
    if frame:
        vol = vol[frame]
    else:
        vol = vol[len(vol) - 1]
    return vol


def model_fn(features, mode, params):
    inputs = features['frames_batch']

    if 'labels_batch' in features:
        labels = features['labels_batch']
    else:
        labels = None

    training = mode == tf.estimator.ModeKeys.TRAIN
    fc_layers = _add_fc_layer(inputs, training)
    logits = _add_regular_rnn_layers(fc_layers, params)

    probabilities = tf.nn.softmax(logits, axis=2, name="softmax_tensor")

    if params['predict_mode'] == 'last':
        logits = tf.map_fn(_get_frame, logits)

        if labels is not None:
            labels = tf.map_fn(_get_frame, labels)

        y_pred = tf.argmax(tf.nn.softmax(logits, axis=1))

    elif params['predict_mode'] == 'mode':
        y_pred = tf.map_fn(_unique_tf, tf.argmax(probabilities, axis=2))
        index_label = tf.map_fn(lambda x: _unique_tf(x, True), tf.argmax(probabilities, axis=2))
        logits = tf.map_fn(lambda x: _slice_integer(x[0], x[1]), (logits, index_label))[0]

        if labels is not None:
            labels = tf.map_fn(lambda x: _slice_integer(x[0], x[1]), (labels, index_label))[0]

    predictions = {
        "classes": tf.one_hot(y_pred, depth=params['classes_amount']),
        "probabilities": probabilities,
        "score": tf.reduce_max(tf.nn.softmax(logits), axis=1),
        "metadata": features['metadata']
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)

    precision = tf.metrics.precision_at_thresholds(labels=labels,
                                                   predictions=tf.nn.softmax(logits),
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


def _slice_integer(vol, int):
    return vol[int], int


if __name__ == '__main__':
    tf.enable_eager_execution()
    logits = tf.random.uniform((8, 15, 101))
    print(logits)
