import numpy as np
import jieba
import joblib
import pickle

X_data = np.load("./data/0_train_X.npy")
Y_data = np.load("./data/0_test_X.npy")
X_data = np.concatenate((X_data, Y_data), axis=0)
X_jieba_cut_data = []
for v_idx, v in enumerate(X_data):
    X_jieba_cut_data.append(jieba.lcut(X_data[v_idx]))

word_set = []
for v in X_jieba_cut_data:
    for j in v:
        word_set.append(j)

word_set = set(word_set)
i = 0
word_dict = {}
for v in word_set:
    word_dict[v] = i
    i += 1
wordVec = joblib.load('/home/bringtree/wordvec/10G_dict.pkl')
word_dict_encoder = {}
for k, v in word_dict.items():
    try:
        word_dict_encoder[k] = wordVec[k]
    except:
        word_dict_encoder[k] = np.ones(400, dtype=np.float32,)
with open('word_dict_encoder.pkl','wb') as fp:
    pickle.dump(word_dict_encoder,fp)