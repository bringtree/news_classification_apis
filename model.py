import tensorflow as tf
import numpy as np
import focal_loss


class Model():
    """
    lstm 模型
    """

    def __init__(self, sentence_len=None, learning_rate=None, word_vec_size=None, hidden_num=None,
                 output_keep_prob=None,
                 batch_size=None):
        """

        :param sentence_len: 句子长度
        :param learning_rate: 学习速率
        :param wordVec_size: 词向量的大小
        :param hidden_num: lstm 隐藏层 size 大小
        :param output_keep_prob: Dropout 层 output 损失 (0,1]
        :param batch_size: batch_size 的大小
        """
        if sentence_len:
            self.sentence_len = sentence_len
        else:
            self.sentence_len = 50
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 0.001
        if word_vec_size:
            self.word_vec_size = word_vec_size
        else:
            self.word_vec_size = 400
        if hidden_num:
            self.hidden_num = hidden_num
        else:
            self.hidden_num = 200
        if output_keep_prob:
            self.output_keep_prob = output_keep_prob
        else:
            self.output_keep_prob = 0.5

    def build(self):
        """
        构建 lstm 模型
        :return: Model对象
        """
        self.input_sentences = tf.placeholder(shape=[None, self.sentence_len, self.wordVec_size],
                                              name='input_sentences',
                                              dtype=tf.float32)
        self.input_labels = tf.placeholder(shape=[None], name='input_labels',
                                           dtype=tf.int32)
        sentence_input = tf.transpose(self.input_sentences, [1, 0, 2])

        self.input_length = tf.placeholder(shape=[None], name='input_length', dtype=tf.int32)

        with tf.variable_scope("bid_lstm_layer"):
            forward_lstm = tf.contrib.rnn.LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
            backward_lstm = tf.contrib.rnn.LSTMCell(self.hidden_num, initializer=tf.orthogonal_initializer())
            forward_drop = tf.contrib.rnn.DropoutWrapper(forward_lstm, output_keep_prob=self.output_keep_prob)
            backward_drop = tf.contrib.rnn.DropoutWrapper(backward_lstm, output_keep_prob=self.output_keep_prob)

            encoder_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=forward_drop,
                cell_bw=backward_drop,
                inputs=sentence_input,
                sequence_length=self.input_length,
                dtype=tf.float32,
                time_major=True)
            lstm_output = tf.concat([encoder_final_state[0].h, encoder_final_state[1].h], 1)

        with tf.variable_scope('full_connection_layer'):
            intent_w = tf.get_variable(
                initializer=tf.random_uniform([self.hidden_num * 2, 2], -0.1, 0.1),
                dtype=tf.float32, name="intent_w")
            intent_b = tf.get_variable(initializer=tf.zeros([2]), dtype=tf.float32, name="intent_b")
            predict_intent = tf.matmul(lstm_output, intent_w) + intent_b

        with tf.variable_scope('predict_result'):
            self.output_intent = tf.argmax(predict_intent, 1, name='output_result')

        with tf.variable_scope("loss_function"):
            cross_entropy = focal_loss.focal_loss(
                prediction_tensor=predict_intent, target_tensor=tf.one_hot(self.input_labels, 2), weights=None,
                alpha=0.25,
                gamma=2
            )
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.variable_scope("optimizer_function"):
            optimizer = tf.train.AdamOptimizer(name="a_optimizer", learning_rate=self.learning_rate)
            self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
            self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)  # clip gradients
            self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

    def train(self, sess, input_sentences, input_labels, input_length):
        """

        :param sess: tf.Session() 返回的对象
        :param input_sentences: 训练的句子 shape = [batch_size, self.sentence_len, self.wordVec_size],
        :param input_labels: 训练的标签 shape = [batch_size]
        :param input_length: 训练的句子长度 shape = [batch_size]
        :return: loss大小 int
        """
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.input_sentences: input_sentences,
            self.input_labels: input_labels,
            self.input_length: input_length
        })
        return loss

    def predict(self, sess, input_sentences, input_length):
        """

        :param sess: tf.Session() 返回的对象
        :param input_sentences: 预测的句子 shape = [batch_size, self.sentence_len, self.wordVec_size],
        :param input_length: 预测的标签 shape = [batch_size]
        :return: 预测的结果 shape = [batch_size]
        """
        result = sess.run([self.output_intent], feed_dict={
            self.input_sentences: input_sentences,
            self.input_length: input_length
        })
        return result
# 测试
# if __name__ == '__main__':
#     sess = tf.Session()
#     lstm_model = Model(sentence_len=50, learning_rate=1, wordVec_size=2, hidden_num=100, output_keep_prob=0.5,
#                        batch_size=3)
#     lstm_model.build()
#     sess.run(tf.global_variables_initializer())
#     s = lstm_model.train(
#         sess,
#         input_sentences=[[[v * 0.1, 1] for v in range(50)],
#                          [[v * 0.2, 1] for v in range(50)],
#                          [[v * 0.3, 1] for v in range(50)]
#                          ],
#         input_labels=[0, 1, 0],
#         input_length=[3, 2, 1]
#     )
#     print(s)
