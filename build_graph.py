import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

if len(sys.argv) != 2:
	sys.exit("Use: python build_graph.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr']
# build corpus
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")




'''
************************************
    Does Not Use Read Pre-trained Word Vectors
************************************
'''
word_embeddings_dim = 50
word_vector_map = {}



'''
************************************
    Read Pre-trained Word Vectors
************************************
'''
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
# _, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])




#******************************************************************************#
#          1.load the train data and test data, and shuffle them               #
#******************************************************************************#
# load the sample name of train(test) data
doc_name_list = []
doc_train_list = []
doc_test_list = []

f = open('data/' + dataset + '.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
    elif temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
f.close()


# load the sample content, the sample order in XXX.clean.txt is the same as that of doc_name_list
doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

# 找出doc_train_list中各条样本在doc_name_list的下标，并存储在train_ids中，然后对train_ids进行打乱
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
random.shuffle(train_ids)

# partial labeled data
#train_ids = train_ids[:int(0.2 * len(train_ids))]

# 将打乱后的train_ids写入文件中
train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

# 找出doc_test_list中各条样本在doc_name_list的下标，并存储在test_ids中，然后对test_ids进行打乱
test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)

random.shuffle(test_ids)

# 将打乱后的test_ids写入文件中
test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

# 变量id存放训练集和测试集的下标
ids = train_ids + test_ids


# 根据train_ids和test_ids中打乱后的顺序，对doc_name_list和doc_content_list重新进行排列，分别得到shuffle_doc_name_list、shuffle_doc_words_list
shuffle_doc_name_list = [] 
shuffle_doc_words_list = []
for id in ids:
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

# 将shuffle_doc_name_list、shuffle_doc_words_list写入文件
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
f = open('data/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_name_str)
f.close()
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
f = open('data/corpus/' + dataset + '_shuffle.txt', 'w')
f.write(shuffle_doc_words_str)
f.close()


#******************************************************************************#
#                           2.build vocabulary                                 #
#******************************************************************************#
word_freq = {}
word_set = set()
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {} # 字典，key=word，value=包含该word的样本的id（该样本在shuffle_doc_words_list中的id）
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {} # 字典，kye=word, value=包含该word的document数量
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {} # 字典，key=word, value=该word在vocab中的下标
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

# 将vocab写入文件
vocab_str = '\n'.join(vocab)
f = open('data/corpus/' + dataset + '_vocab.txt', 'w')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''









#******************************************************************************#
#                           3.记录label的种类                                   #
#******************************************************************************#
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label_set.add(temp[2])
label_list = list(label_set)

# 将label_list写入文件
label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w')
f.write(label_list_str)
f.close()











#******************************************************************************#
#        4. divide train_set into 90% real_train_set and 10% val_set           #
#******************************************************************************#
# 从train_ids选择90%作为真正的训练集，剩下的10%作为验证集
train_size = len(train_ids)
val_size = int(0.1 * train_size)
real_train_size = train_size - val_size  

# 真正参与训练的训练集的样本名称
real_train_doc_names = shuffle_doc_name_list[:real_train_size]

# 将真正参与训练的训练集的样本名称写入文件
real_train_doc_names_str = '\n'.join(real_train_doc_names)
f = open('data/' + dataset + '.real_train.name', 'w')
f.write(real_train_doc_names_str)
f.close()



#*******************************************************************************************************#
#    5. 构造模型能够处理的训练集数据
#     x: 矩阵，大小为real_train_size*word_embeddings_dim，每行代表一个document的embedding。 
#        如果不使用预训练的词向量，则该document的embedding为0。如果使用预训练的词向量，则该
#        document的embedding为该document中各个单词embedding之和。
#     y: 矩阵，大小为real_train_size*len(label_list)，每一行代表一个document对应label的one_hot向量
#*******************************************************************************************************#

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map: # word_vector_map，字典，是预训练词向量矩阵
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
# print(y)







#*************************************************************************************************#
#    6. 构造模型能够处理的测试集数据。
#    tx: 矩阵，大小为test_size*word_embeddings_dim，每行代表一个document的embedding。 
#       如果不使用预训练的词向量，则该document的embedding为0。如果使用预训练的词向量，则该
#       document的embedding为该document中各个单词embedding之和。
#    ty: 矩阵，大小为test_size*len(label_list)，每一行代表一个document对应label的one_hot向量
#*************************************************************************************************#

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
# print(ty)




#************************************************************************************************************************************#
#    7. 构建以下变量：
#    allx: 矩阵, 大小为(train_size + vocab_size)*word_embeddings_dim, 前train_size行存放document的embedding，后
#          vocab_size行存放单词表各个单词的词向量。如果不使用预训练的词向量，词向量默认为随机初始化。
#    ally: 矩阵，大小为(train_size + vocab_size)*label_list, 前train_size行存放document的label的one_hot编码，后vocab_size行都存放0向量
#************************************************************************************************************************************#

# allx: the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)
    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len


for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))


row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)








#*************************************************************************************************************
#                      8. build the Doc word heterogeneous graph (adjacent matrix)
#*************************************************************************************************************

# word co-occurence with context windows
window_size = 20
windows = []

'''
    以大小为window_size、滑动步长为1的滑动窗口对每个document的单词进行分组，分组结果存入列表windows中。
    例如windows[0]=['organization','university','maine'], windows[1]=['pin','map','din','cable']
'''
for doc_words in shuffle_doc_words_list:
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

'''
    word_window_freq, 字典, key=单词，value=包含该单词的window的个数
'''
word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

'''
    以每个滑动窗口内的单词分组结果为单位，统计两个不同单词间所能产生的组合个数，
    并存入字典word_pair_count，其key=单词1和单词2的id组合，value=包含该组合的window的数量（在window范围内，出现的共现次数）
    
    例如第一层循环时，取得的window共有5个单词，每个单词在列表window中的位置为：window=[0,1,2,3,4,5]
    那么第二层和第三层的循环，其需要处理列表window中以下两个位置单词组合的情况：
    j-i
    0-1
    0-2
    1-2
    0-3
    1-3
    2-3
    0-4
    1-4
    2-4
    3-4
    0-5
    1-5
    2-5
    3-5
    4-5
'''
word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            # 第一类组合: a-b
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # 第二类组合：b-a
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1



'''
    计算单词和单词之间的边的权重pmi
'''
row = []
col = []
weight = []
num_window = len(windows)

for key in word_pair_count:
    temp = key.split(',')
    i = int(temp[0]) # 单词1的id
    j = int(temp[1]) # 单词2的id
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''


'''
    doc_word_freq，字典，key="str(doc_id) + ',' + str(word_id)", value=该doc_id和该word_id共现的次数
'''
# doc word frequency
doc_word_freq = {}
for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1


'''
    计算单词和文档之间的边的权重tf_idf
'''
for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size: # 训练集部分
            row.append(i)
        else: # 测试集部分
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) / word_doc_freq[vocab[j]])
        tf_idf = freq * idf
        weight.append(tf_idf)
        doc_word_set.add(word)

node_size = train_size + vocab_size + test_size
adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size)) # 邻接矩阵



#***************************************************************************
#                 9. save the object: x,y,tx,ty,allx,ally,adj
#***************************************************************************
'''
    dump objects
'''
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()
