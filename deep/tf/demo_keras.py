#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : demo_keras.py
# @Author: sl
# @Date  : 2021/10/31 - 上午9:46

"""
多输入与多输出网络
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


def multi_input_output():
    """
    构建一个根据文档内容、标签和标题，预测文档优先级和执行部门的网络
    :return:
    """
    # 超参
    num_words = 2000
    num_tags = 12
    num_departments = 4
    # 输入
    body_input = keras.Input(shape=(None,), name='body')
    title_input = keras.Input(shape=(None,), name='title')
    tag_input = keras.Input(shape=(num_tags,), name='tag')
    # 嵌入层
    body_feat = layers.Embedding(num_words, 64)(body_input)
    title_feat = layers.Embedding(num_words, 64)(title_input)
    # 特征提取层
    body_feat = layers.LSTM(32)(body_feat)
    title_feat = layers.LSTM(128)(title_feat)
    features = layers.concatenate([title_feat, body_feat, tag_input])
    # 分类层
    priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(features)
    department_pred = layers.Dense(num_departments, activation='softmax', name='department')(features)
    # 构建模型
    model = keras.Model(inputs=[body_input, title_input, tag_input],
                        outputs=[priority_pred, department_pred])
    model.summary()
    keras.utils.plot_model(model, 'multi_model.png', show_shapes=True)
    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss={'priority': 'binary_crossentropy',
                        'department': 'categorical_crossentropy'},
                  loss_weights=[1., 0.2])
    # 载入输入数据
    title_data = np.random.randint(num_words, size=(1280, 10))
    body_data = np.random.randint(num_words, size=(1280, 100))
    tag_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
    # 标签
    priority_label = np.random.random(size=(1280, 1))
    department_label = np.random.randint(2, size=(1280, num_departments))
    # 训练
    history = model.fit(
        {'title': title_data, 'body': body_data, 'tag': tag_data},
        {'priority': priority_label, 'department': department_label},
        batch_size=32,
        epochs=5
    )


def small_resnet():
    inputs = keras.Input(shape=(32, 32, 3), name='img')
    h1 = layers.Conv2D(32, 3, activation='relu')(inputs)
    h1 = layers.Conv2D(64, 3, activation='relu')(h1)
    block1_out = layers.MaxPooling2D(3)(h1)

    h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(block1_out)
    h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(h2)
    block2_out = layers.add([h2, block1_out])

    h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(block2_out)
    h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(h3)
    block3_out = layers.add([h3, block2_out])

    h4 = layers.Conv2D(64, 3, activation='relu')(block3_out)
    h4 = layers.GlobalMaxPool2D()(h4)
    h4 = layers.Dense(256, activation='relu')(h4)
    h4 = layers.Dropout(0.5)(h4)
    outputs = layers.Dense(10, activation='softmax')(h4)

    model = keras.Model(inputs, outputs, name='small resnet')
    model.summary()
    keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = y_train.astype('float32') / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model.compile(optimizer=keras.optimizers.RMSprop(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.fit(x_train, y_train,
              batch_size=64,
              epochs=1,
              validation_split=0.2)

    model.predict(x_test, batch_size=32)


def demo_mnist():
    # 模型构造
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = layers.Dense(64, activation='relu')(inputs)
    h1 = layers.Dense(64, activation='relu')(h1)
    outputs = layers.Dense(10, activation='softmax')(h1)
    model = keras.Model(inputs, outputs)
    # keras.utils.plot_model(model, 'net001.png', show_shapes=True)

    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])

    # 载入数据
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255

    x_val = x_train[-10000:]
    y_val = y_train[-10000:]

    x_train = x_train[:-10000]
    y_train = y_train[:-10000]

    # 训练模型
    history = model.fit(x_train, y_train, batch_size=64, epochs=3,
                        validation_data=(x_val, y_val))
    print('history:')
    print(history.history)

    result = model.evaluate(x_test, y_test, batch_size=128)
    print('evaluate:')
    print(result)
    pred = model.predict(x_test[:2])
    print('predict:')
    print(pred)


class CatgoricalTruePostives(keras.metrics.Metric):
    def __init__(self, name='binary_true_postives', **kwargs):
        super(CatgoricalTruePostives, self).__init__(name=name, **kwargs)
        self.true_postives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred)
        y_true = tf.equal(tf.cast(y_pred, tf.int32), tf.cast(y_true, tf.int32))

        y_true = tf.cast(y_true, tf.float32)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            y_true = tf.multiply(sample_weight, y_true)

        return self.true_postives.assign_add(tf.reduce_sum(y_true))

    def result(self):
        return tf.identity(self.true_postives)

    def reset_states(self):
        self.true_postives.assign(0.)


# 以定义网络层的方式添加网络loss
class ActivityRegularizationLayer(layers.Layer):
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs) * 0.1)
        return inputs


# 也可以以定义网络层的方式添加要统计的metric
class MetricLoggingLayer(layers.Layer):
    def call(self, inputs):
        self.add_metric(keras.backend.std(inputs),
                        name='std_of_activation',
                        aggregation='mean')

        return inputs


def get_compiled_model():
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = layers.Dense(64, activation='relu')(inputs)
    h2 = layers.Dense(64, activation='relu')(h1)
    outputs = layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


def get_compiled_model():
    inputs = keras.Input(shape=(784,), name='mnist_input')
    h1 = layers.Dense(64, activation='relu')(inputs)
    h2 = layers.Dense(64, activation='relu')(h1)
    outputs = layers.Dense(10, activation='softmax')(h2)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[keras.metrics.SparseCategoricalAccuracy()])
    return model


def demo_data():
    model = get_compiled_model()

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    # model.fit(train_dataset, epochs=3)
    # steps_per_epoch 每个epoch只训练几步
    # validation_steps 每次验证，验证几步
    model.fit(train_dataset, epochs=3, steps_per_epoch=100,
              validation_data=val_dataset, validation_steps=3)


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255
    x_test = x_test.reshape(10000, 784).astype('float32') / 255
    x_val = x_train[-10000:]
    y_val = y_train[-10000:]
    x_train = x_train[:-10000]
    y_train = y_train[:-10000]
    return x_train, y_train, x_val, y_val, x_test, y_test


def demo_data_weight():
    # 类权重

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()

    model = get_compiled_model()
    class_weight = {i: 1.0 for i in range(10)}
    class_weight[5] = 2.0
    print(class_weight)
    model.fit(x_train, y_train,
              class_weight=class_weight,
              batch_size=64,
              epochs=4)
    # 样本权重
    model = get_compiled_model()
    sample_weight = np.ones(shape=(len(y_train),))
    sample_weight[y_train == 5] = 2.0
    model.fit(x_train, y_train,
              sample_weight=sample_weight,
              batch_size=64,
              epochs=4)
    # tf.data数据
    model = get_compiled_model()

    sample_weight = np.ones(shape=(len(y_train),))
    sample_weight[y_train == 5] = 2.0

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train,
                                                        sample_weight))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    model.fit(train_dataset, epochs=3, )


def multi_in_out():
    """
    多输入多输出模型
    :return:
    """
    image_input = keras.Input(shape=(32, 32, 3), name='img_input')
    timeseries_input = keras.Input(shape=(None, 10), name='ts_input')

    x1 = layers.Conv2D(3, 3)(image_input)
    x1 = layers.GlobalMaxPooling2D()(x1)

    x2 = layers.Conv1D(3, 3)(timeseries_input)
    x2 = layers.GlobalMaxPooling1D()(x2)

    x = layers.concatenate([x1, x2])

    score_output = layers.Dense(1, name='score_output')(x)
    class_output = layers.Dense(5, activation='softmax', name='class_output')(x)

    model = keras.Model(inputs=[image_input, timeseries_input],
                        outputs=[score_output, class_output])
    keras.utils.plot_model(model, 'multi_input_output_model.png'
                           , show_shapes=True)
    # 可以为模型指定不同的loss和metrics
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[keras.losses.MeanSquaredError(),
              keras.losses.CategoricalCrossentropy()])

    # 还可以指定loss的权重
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={'score_output': keras.losses.MeanSquaredError(),
              'class_output': keras.losses.CategoricalCrossentropy()},
        metrics={'score_output': [keras.metrics.MeanAbsolutePercentageError(),
                                  keras.metrics.MeanAbsoluteError()],
                 'class_output': [keras.metrics.CategoricalAccuracy()]},
        loss_weight={'score_output': 2., 'class_output': 1.})

    # 可以把不需要传播的loss置0
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[None, keras.losses.CategoricalCrossentropy()])

    # Or dict loss version
    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss={'class_output': keras.losses.CategoricalCrossentropy()})


def call_back():
    """
    回调使用
    :return:
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()

    model = get_compiled_model()

    callbacks = [
        keras.callbacks.EarlyStopping(
            # 是否有提升关注的指标
            monitor='val_loss',
            # 不再提升的阈值
            min_delta=1e-2,
            # 2个epoch没有提升就停止
            patience=2,
            verbose=1)
    ]
    model.fit(x_train, y_train,
              epochs=20,
              batch_size=64,
              callbacks=callbacks,
              validation_split=0.2)
    # checkpoint模型回调
    model = get_compiled_model()
    check_callback = keras.callbacks.ModelCheckpoint(
        filepath='mymodel_{epoch}.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    model.fit(x_train, y_train,
              epochs=3,
              batch_size=64,
              callbacks=[check_callback],
              validation_split=0.2)
    # 动态调整学习率
    initial_learning_rate = 0.1
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = keras.optimizers.RMSprop(learning_rate=lr_schedule)
    # 使用tensorboard
    tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='./full_path_to_your_logs')
    model.fit(x_train, y_train,
              epochs=5,
              batch_size=64,
              callbacks=[tensorboard_cbk],
              validation_split=0.2)


# 自定义callback
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.losses = []

    def on_epoch_end(self, batch, logs):
        self.losses.append(logs.get('loss'))
        print('\nloss:', self.losses[-1])


def user_define_train():
    """
    自定义训练
    :return:
    """
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()

    # 构建一个全连接网络.
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # 优化器.
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # 损失函数.
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    # 准备数据.
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # 自己构造循环
    for epoch in range(3):
        print('epoch: ', epoch)
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            # 开一个gradient tape, 计算梯度
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)

                loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))


def user_define_train_and_valid():
    """
    自定义训练 和验证
    :return:
    """

    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()

    # 训练并验证
    # 获取模型
    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # sgd优化器
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    # 分类损失函数
    loss_fn = keras.losses.SparseCategoricalCrossentropy()

    # 设定统计参数
    train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # 准备训练数据
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # 准备验证数据
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    # 迭代训练
    for epoch in range(3):
        print('Start of epoch %d' % (epoch,))

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 更新统计传输
            train_acc_metric(y_batch_train, logits)

            # 输出
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))

        # 输出统计参数的值
        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        # 重置统计参数
        train_acc_metric.reset_states()

        # 用模型进行验证
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            # 根据验证的统计参数
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print('Validation acc: %s' % (float(val_acc),))


def user_define_loss():
    x_train, y_train, x_val, y_val, x_test, y_test = get_mnist_data()
    # 准备训练数据
    batch_size = 64
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # 准备验证数据
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(64)

    inputs = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    # Insert activity regularization as a layer
    x = ActivityRegularizationLayer()(x)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    logits = model(x_train[:64])
    print(model.losses)
    logits = model(x_train[:64])
    logits = model(x_train[64: 128])
    logits = model(x_train[128: 192])
    print(model.losses)
    # 将loss添加进求导中
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)

    # 损失函数.
    loss_fn = keras.losses.SparseCategoricalCrossentropy()


    for epoch in range(3):
        print('Start of epoch %d' % (epoch,))

        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                loss_value = loss_fn(y_batch_train, logits)

                # 添加额外的loss
                loss_value += sum(model.losses)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # 每200个batch输出一次学习.
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))


if __name__ == '__main__':
    # multi_input_output()

    # small_resnet()
    demo_data_weight()
