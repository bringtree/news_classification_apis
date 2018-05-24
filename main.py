import sys
import os
import tensorflow as tf
import model
import joblib
import numpy as np
import jieba
from tensorflow.python.framework import graph_util
import configparser


# export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


def make_batches(x_src, y_src, batch_size):
    """
    生成 batches
    :param x_src: 输入的新闻标题的numpy路径
    :param y_src:  输入的标签的numpy路径
    :param batch_size: batch_size 设置的大小
    :return: 生成的新闻标题的batches,生成标签的batches以及batches的数量
    """
    x = np.load(x_src)
    y = np.load(y_src)

    cut_x = [jieba.lcut(v) for v in x]

    batches_num = int(len(x) / batch_size)

    data_x_batches = [[] for i in range(batches_num)]
    data_y_batches = [[] for i in range(batches_num)]

    # 先填充 标记为1(食品安全的) 的 新闻
    i = 0
    for idx, v in enumerate(y):
        if v == 1:
            data_x_batches[i % batches_num].append(cut_x[idx])
            data_y_batches[i % batches_num].append(y[idx])
            i += 1
    # 先填充 标记为0(食品不安全的) 的 新闻
    i = 0
    for idx, v in enumerate(y):
        if v == 0:
            if len(data_y_batches[-1]) == batch_size:
                break
            while len(data_x_batches[i % batches_num]) == batch_size:
                i += 1
            data_x_batches[i % batches_num].append(cut_x[idx])
            data_y_batches[i % batches_num].append(y[idx])
    return data_x_batches, data_y_batches, batches_num


def encoder_data(batches_num, x_batches, y_batches, word_vec, word_vec_len, word_vec_size):
    """
    对输入的新闻标题进行编码，和数据格式整理的扩充
    :param batches_num: batches_num的数目
    :param x_batches: 未编码前 新闻标题 的batches
    :param y_batches: 未格式化前 标签 的batches
    :param word_vec: 用来编码的字典
    :param word_vec_len: 新闻标题的最大长度
    :param word_vec_size: 编码的词向量的字典的大小
    :return:
    """
    data_x = []
    data_y = []
    data_len = []
    for i in range(batches_num):
        x = x_batches[i]
        y = y_batches[i]
        data_len.append([len(sentence) for sentence in x])
        encoder_x = []
        for sentence in x:
            tmp = []
            for word in sentence:
                tmp.append(word_vec[word])
            while len(tmp) < word_vec_len:
                tmp.append(np.zeros(word_vec_size, dtype=np.float32))
            tmp = np.array(tmp)
            encoder_x.append(tmp)
        encoder_x = np.array(encoder_x)
        data_x.append(encoder_x)
        data_y.append(y)
    return data_x, data_y, data_len


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('./params.cfg')
    batch_size = config.get('hyperparameters', 'batch_size')
    save_file_num = config.get('hyperparameters', 'save_file_num')
    word_vec_size = config.get('hyperparameters', 'word_vec_size')
    word_vec_len = config.get('hyperparameters', 'word_vec_len')
    iter_num = config.get('hyperparameters', 'iter_num')
    base_acc = config.get('plugin', 'base_acc')
    base_f1_score = config.get('plugin', 'base_f1_score')
    save_pb_mode = config.get('plugin', 'save_pb_mode')
    pb_file = config.get('data', 'pb_filepath')
    model_src = config.get('data', 'model_filepath')
    word_vec_path = config.get('data', 'word_vec_filepath')
    word_vec = joblib.load(word_vec_path)

    # 数据预处理
    train_x_batches, train_y_batches, train_batches_num = make_batches("0_train_X.npy", "0_train_Y.npy", batch_size)
    train_x, train_y, train_len = encoder_data(train_batches_num, train_x_batches, train_y_batches, word_vec,
                                               word_vec_len,
                                               word_vec_size)
    test_x_batches, test_y_batches, test_batches_num = make_batches("0_test_X.npy", "0_test_Y.npy", batch_size)
    test_x, test_y, test_len = encoder_data(test_batches_num, test_x_batches, test_y_batches, word_vec, word_vec_len,
                                            word_vec_size)
    ###############################################################

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    lstm_model = model.Model()
    lstm_model.build()
    sess.run(tf.global_variables_initializer())

    # 模型加载和保存
    saver = tf.train.Saver(max_to_keep=save_file_num)
    ckpt = tf.train.get_checkpoint_state(model_src)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    if save_pb_mode is True:
        constant_graph = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            ['output_result'],
            variable_names_whitelist=None,
            variable_names_blacklist=None
        )
        with tf.gfile.FastGFile('./' + pb_file, mode='wb') as f:
            f.write(constant_graph.SerializeToString())

    for iter in range(iter_num):
        loss, right_num, err_num, TP, FP, TN, FN, = 0, 0, 0, 0, 0, 0, 0
        for i in range(train_batches_num):
            loss += lstm_model.train(
                sess,
                input_sentences=train_x[i],
                input_labels=train_y[i],
                input_length=train_len[i]
            )
        for i in range(test_batches_num):
            result = lstm_model.predict(
                sess,
                input_sentences=test_x[i],
                input_length=test_len[i]
            )
            # 计算f1 分数
            result = result[0]
            for idx, v in enumerate(result):
                if result[idx] == test_y[i][idx]:
                    if test_y[i][idx] == 1:
                        TP += 1
                    else:
                        TN += 1
                    right_num += 1
                else:
                    if test_y[i][idx] == 1:
                        FP += 1
                    else:
                        FN += 1
                    err_num += 1
        precision = TP / (TP + FN + 1e-5)
        recall = TP / (TP + FP + 1e-5)
        acc = right_num / (err_num + right_num)
        f1_score = 2 * precision * recall / (precision + recall + 1e-5)

        # 保存模型
        if (acc > base_acc) and (f1_score > base_f1_score):
            if not os.path.exists(model_src):
                os.makedirs(model_src)
            saver.save(sess, model_src, global_step=iter)

        print('acc: ' + str(acc),
              "F1:" + str(f1_score),
              'loss' + str(loss))
