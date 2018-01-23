import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

'''
Website:http://cuijiahua.linear_com
Modify:2018-01-23
Author:Jack Cui
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.33)
config.gpu_options.allow_growth = True


max_steps = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.9   # dropout时随机保留神经元的比例
data_dir = './MNIST_DATA'   # 样本数据存储的路径
log_dir = './MNIST_LOG'    # 输出日志保存的路径

# 获取数据集，并采用采用one_hot热编码
mnist = input_data.read_data_sets(data_dir,one_hot = True)

sess = tf.InteractiveSession(config = config)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

# 保存图像信息
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

# 初始化权重参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

# 初始化偏执参数
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# 绘制参数变化
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)

# 构建神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 设置命名空间
    with tf.name_scope(layer_name):
        # 调用之前的方法初始化权重w，并且调用参数信息的记录方法，记录w的信息
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        # 调用之前的方法初始化权重b，并且调用参数信息的记录方法，记录b的信息
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        # 执行wx+b的线性计算，并且用直方图记录下来
        with tf.name_scope('linear_compute'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('linear', preactivate)
        # 将线性输出经过激励函数，并将输出也用直方图记录下来
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
    # 返回激励层的最终输出
    return activations
        
hidden1 = nn_layer(x, 784, 500, 'layer1')

# 创建dropout层
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)
    
    
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

# 创建损失函数
with tf.name_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    with tf.name_scope('total'):
        # 计算所有样本交叉熵损失的均值
        cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('loss', cross_entropy)
    
    
# 使用AdamOptimizer优化器训练模型，最小化交叉熵损失
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 计算准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        

tf.summary.scalar('accuracy', accuracy)

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
tf.global_variables_initializer().run()

def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

for i in range(max_steps):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()