from flask import Flask
import tensorflow as tf
import jieba
import numpy as np
import joblib
from flask import request,jsonify

app = Flask(__name__)
# app.debug = True

# wordVec = joblib.load("./word_dict_encoder.pkl")
# wordVec = joblib.load("/home/10G_dict.pkl")
wordVec = joblib.load("/home/bringtree/wordvec/10G_dict.pkl")
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with open('./test.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    op = sess.graph.get_tensor_by_name('output_result:0')
    input_sen = sess.graph.get_tensor_by_name('input_sentences:0')
    input_len = sess.graph.get_tensor_by_name('input_length:0')


@app.route('/news_recognition', methods=['POST'])
def new_recognition():
    res_info = {
        'code': '-1',
    }
    if request.form['news_title'] != '':
        sentence = request.form['news_title']
        sentence = jieba.lcut(sentence)
        sentence_len = len(sentence)
        sentence_encoder = []
        for v in sentence:
            if v in wordVec:
                sentence_encoder.append(wordVec[v])
            else:
                sentence_encoder.append(np.ones(400, dtype=np.float32))
        while len(sentence_encoder) < 50:
            sentence_encoder.append(np.zeros(400, dtype=np.float32))
        pre = sess.run(op, {
            input_sen: [sentence_encoder],
            input_len: [sentence_len]
        })

        res_info['code'] = str(pre[0])

    return jsonify(res_info)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
