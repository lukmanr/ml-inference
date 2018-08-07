import time
import os
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},y=train_labels,batch_size=100,num_epochs=None,shuffle=True)
eval_input_fn  = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},y=eval_labels,num_epochs=1,shuffle=False)
pred_input_fn  = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},num_epochs=1,shuffle=False)


# ## 3. Create Estimator

mnist_path = './mnist/'

# Convolution Block

def _conv(x,kernel,name,log=False):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal(shape=kernel,stddev=0.01),name='W')
        b = tf.Variable(tf.constant(0.0,shape=[kernel[3]]),name='b')
        conv = tf.nn.conv2d(x, W, strides=[1,1,1,1],padding='SAME')
        activation = tf.nn.relu(tf.add(conv,b))
        pool = tf.nn.max_pool(activation,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        if log==True:
            tf.summary.histogram("weights",W)
            tf.summary.histogram("biases",b)
            tf.summary.histogram("activations",activation)
        return pool

# Dense Block

def _dense(x,size_in,size_out,name,relu=False,log=False):
    with tf.name_scope(name):
        flat = tf.reshape(x,[-1,size_in])
        W = tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name='W')
        b = tf.Variable(tf.constant(0.0,shape=[size_out]),name='b')
        activation = tf.add(tf.matmul(flat,W),b)
        if relu==True:
            activation = tf.nn.relu(activation)
        if log==True:
            tf.summary.histogram("weights",W)
            tf.summary.histogram("biases",b)
            tf.summary.histogram("activations",activation)
        return activation


def cnn_model_fn(features, labels, mode, params):

    #### 1 INFERENCE MODEL

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = _conv(input_layer,kernel=[5,5,1,64],name='conv1',log=params['log'])
    conv2 = _conv(conv1,kernel=[5,5,64,64],name='conv2',log=params['log'])
    dense = _dense(conv2,size_in=7*7*64,size_out=params['dense_units'],
                   name='Dense',relu=True,log=params['log'])
    if mode==tf.estimator.ModeKeys.TRAIN:
        dense = tf.nn.dropout(dense,params['drop_out'])
    logits = _dense(dense,size_in=params['dense_units'],
                    size_out=10,name='Output',relu=False,log=params['log'])

    #### 2 CALCULATIONS AND METRICS

    predictions = {"classes": tf.argmax(input=logits,axis=1),
                   "logits": logits,
                   "probabilities": tf.nn.softmax(logits,name='softmax')}
    export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
    if (mode==tf.estimator.ModeKeys.TRAIN or mode==tf.estimator.ModeKeys.EVAL):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
        accuracy = tf.metrics.accuracy(
            labels=labels, predictions=tf.argmax(logits,axis=1))
        metrics = {'accuracy':accuracy}

    #### 3 MODE = PREDICT

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions, export_outputs=export_outputs)

    #### 4 MODE = TRAIN

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            params['learning_rate'],tf.train.get_global_step(),
            decay_steps=100000,decay_rate=0.96)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        if params['replicate']==True:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('accuracy',accuracy[1])
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    #### 5 MODE = EVAL

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,loss=loss,eval_metric_ops=metrics)


# ###### Set paramters and model path

config     = tf.estimator.RunConfig(save_checkpoints_secs = 30, keep_checkpoint_max = 5)

mnist_path = './mnist/'


model_params  = {'drop_out'      : 0.8,
                 'dense_units'   : 1024,
                 'learning_rate' : 1e-3,
                 'log'           : True,
                 'replicate'     : False
                }

name = 'cnn_model_'


if model_params['replicate']==True:
    cnn_model_fn = tf.contrib.estimator.replicate_model_fn(
        cnn_model_fn, loss_reduction=tf.losses.Reduction.MEAN)

name = 'cnn_model/cnn_model_'
if model_params['replicate']==True:
    name = 'cnn_model_dist/cnn_model_'
name = name + 'dense(' + str(model_params['dense_units']) + ')_'
name = name + 'drop(' + str(model_params['drop_out']) + ')_'
name = name + 'lr(' + str(model_params['learning_rate']) + ')_'
name = name + time.strftime("%Y%m%d%H%M%S")
cnn_dir  = os.path.join(mnist_path,name)

print(cnn_dir)

cnn_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,model_dir=cnn_dir,params=model_params,config=config)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=20000)
eval_spec  = tf.estimator.EvalSpec(input_fn=eval_input_fn,throttle_secs=10)

tf.estimator.train_and_evaluate(cnn_classifier, train_spec, eval_spec)

predictions = list(cnn_classifier.predict(input_fn=pred_input_fn))
print(predictions[0])


def serving_input_receiver_fn():
    receiver_tensor = {'x': tf.placeholder(shape=[None,28,28,1], dtype=tf.float32)}
    features = {'x': receiver_tensor['x']}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

cnn_classifier.export_savedmodel(
    export_dir_base=cnn_dir, serving_input_receiver_fn=serving_input_receiver_fn)

