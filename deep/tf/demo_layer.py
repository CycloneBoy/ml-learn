#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo_layer.py
# @Author: sl
# @Date  : 2021/10/31 - 上午10:55

"""
自定义层

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


# 定义网络层就是：设置网络权重和输出到输入的计算过程
class MyLayer(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer, self).__init__()

        w_init = tf.random_normal_initializer()
        self.weight = tf.Variable(initial_value=w_init(
            shape=(input_dim, unit), dtype=tf.float32), trainable=True)

        b_init = tf.zeros_initializer()
        self.bias = tf.Variable(initial_value=b_init(
            shape=(unit,), dtype=tf.float32), trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


class MyLayer2(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer2, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias



def demo_layer():
    x = tf.ones((3, 5))
    my_layer = MyLayer2(5, 4)
    out = my_layer(x)
    print(out)


class AddLayer(layers.Layer):
    def __init__(self, input_dim=32):
        super(AddLayer, self).__init__()
        self.sum = self.add_weight(shape=(input_dim,),
                                   initializer=keras.initializers.Zeros(),
                                   trainable=False)

    def call(self, inputs):
        self.sum.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.sum


def demo_2():
    x = tf.ones((3, 3))
    my_layer = AddLayer(3)
    out = my_layer(x)
    print(out.numpy())
    out = my_layer(x)
    print(out.numpy())
    print('weight:', my_layer.weights)
    print('non-trainable weight:', my_layer.non_trainable_weights)
    print('trainable weight:', my_layer.trainable_weights)


class MyLayer3(layers.Layer):
    def __init__(self, unit=32):
        super(MyLayer3, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

def demo_4():
    my_layer = MyLayer3(3)
    x = tf.ones((3, 5))
    out = my_layer(x)
    print(out)
    my_layer = MyLayer3(3)

    x = tf.ones((2, 2))
    out = my_layer(x)
    print(out)


class Linear(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(Linear, self).get_config()
        config.update({'units': self.units})
        return config


def demo_save_1():
    layer = Linear(125)
    config = layer.get_config()
    print(config)
    new_layer = Linear.from_config(config)


class MyDropout(layers.Layer):
    def __init__(self, rate, **kwargs):
        super(MyDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.cond(training,
                       lambda: tf.nn.dropout(inputs, rate=self.rate),
                       lambda: inputs)


# 采样网络
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# 编码器
class Encoder(layers.Layer):
    def __init__(self, latent_dim=32,
                 intermediate_dim=64, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_mean = layers.Dense(latent_dim)
        self.dense_log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        z_mean = self.dense_mean(h1)
        z_log_var = self.dense_log_var(h1)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


# 解码器
class Decoder(layers.Layer):
    def __init__(self, original_dim,
                 intermediate_dim=64, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu')
        self.dense_output = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        h1 = self.dense_proj(inputs)
        return self.dense_output(h1)


# 变分自编码器
class VAE(tf.keras.Model):
    def __init__(self, original_dim, latent_dim=32,
                 intermediate_dim=64, name='encoder', **kwargs):
        super(VAE, self).__init__(name=name, **kwargs)

        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_sum(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(kl_loss)
        return reconstructed


def train_vae():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    vae = VAE(784, 32, 64)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
    vae.fit(x_train, x_train, epochs=3, batch_size=64)


def user_train_vae():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
    
    original_dim = 784
    vae = VAE(original_dim, 64, 32)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    mse_loss_fn = tf.keras.losses.MeanSquaredError()
    
    loss_metric = tf.keras.metrics.Mean()
    
    # 每个epoch迭代.
    for epoch in range(3):
        print('Start of epoch %d' % (epoch,))
    
    # 取出每个batch的数据并训练.
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # 计算 reconstruction loss
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # 添加 KLD regularization loss

        grads = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(grads, vae.trainable_variables))

        loss_metric(loss)

        if step % 100 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result()))


def build_model_to_save():
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')
    model.summary()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop())
    history = model.fit(x_train, y_train,
                        batch_size=64,
                        epochs=1)

    predictions = model.predict(x_test)

    model.save('the_save_model.h5')
    new_model = keras.models.load_model('the_save_model.h5')
    new_prediction = new_model.predict(x_test)
    np.testing.assert_allclose(predictions, new_prediction, atol=1e-6)  # 预测结果一样


def show_loss():
    # 训练一个模型
    (mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

    dataset = tf.data.Dataset.from_tensor_slices(
      (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
       tf.cast(mnist_labels,tf.int64)))
    dataset = dataset.shuffle(1000).batch(32)
    mnist_model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                             input_shape=(None, None, 1)),
      tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
      tf.keras.layers.GlobalAveragePooling2D(),
      tf.keras.layers.Dense(10)
    ])
    for images,labels in dataset.take(1):
        print("Logits: ", mnist_model(images[0:1]).numpy())

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_history = []
    for (batch, (images, labels)) in enumerate(dataset.take(400)):
        if batch % 10 == 0:
            print('.', end='')
        with tf.GradientTape() as tape:
            logits = mnist_model(images, training=True)
            loss_value = loss_object(labels, logits)

        loss_history.append(loss_value.numpy().mean())
        grads = tape.gradient(loss_value, mnist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))
    import matplotlib.pyplot as plt

    plt.plot(loss_history)
    plt.xlabel('Batch #')
    plt.ylabel('Loss [entropy]')


def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y


def mnist_dataset():
  (x, y), _ = tf.keras.datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds




def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


@tf.function
def train(model, optimizer):
  train_ds = mnist_dataset()
  step = 0
  loss = 0.0
  accuracy = 0.0
  for x, y in train_ds:
    step += 1
    loss = train_one_step(model, optimizer, x, y)
    if tf.equal(step % 10, 0):
      tf.print('Step', step, ': loss', loss, '; accuracy', compute_accuracy.result())
  return step, loss, accuracy

def use_tf_function():
    pass

if __name__ == '__main__':
    pass
    # demo_layer()
    # demo_2()
    # demo_4()
    # demo_save_1()

    train_dataset = mnist_dataset()
    model = tf.keras.Sequential((
        tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10)))
    model.build()
    optimizer = tf.keras.optimizers.Adam()
    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    step, loss, accuracy = train(model, optimizer)


