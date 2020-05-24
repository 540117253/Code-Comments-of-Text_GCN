import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    # training nodes are training docs, no initial features
    # print("x: ", x)
    # test nodes are training docs, no initial features
    # print("tx: ", tx)
    # both labeled and unlabeled training instances are training docs and words
    # print("allx: ", allx)
    # training labels are training doc labels
    # print("y: ", y)
    # test labels are test doc labels
    # print("ty: ", ty)
    # ally are labels for labels for allx, some will not have labels, i.e., all 0
    # print("ally: \n")
    # for i in ally:
    # if(sum(i) == 0):
    # print(i)
    # graph edge weight is the word co-occurence or doc word frequency
    # no need to build map, directly build csr_matrix
    # print('graph : ', graph)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    return:
        adj, 稀疏矩阵，原本预处理的非对称矩阵adj进行对称化后的矩阵
        features, 矩阵，大小为(train_size+vocab_size+test_size)*word_embeddings_dim，矩阵allx和矩阵tx进行纵向拼接后的矩阵
        y_train, 矩阵，大小为(train_y+vocab+test_y)*len(label_list)，根据训练集样本的下标idx_train，在矩阵相应的行存放该样本的label的one-hot编码，其余行为0向量
        y_val, 矩阵，大小为(train_y+vocab+test_y)*len(label_list)，根据验证集样本的下标idx_val，在矩阵相应的行存放该样本的label的one-hot编码，，其余行为0向量
        y_test, 矩阵，大小为(train_y+vocab+test_y)*len(label_list)，根据测试集样本的下标idx_test，在矩阵相应的行存放该样本的label的one-hot编码，，其余行为0向量
        train_mask, bool向量，长度为(train_size+vocab_size+test_size)，将train_size中的前边部分（real_train_size）的单元标为True，剩下单元标为False
        val_mask, bool向量，长度为(train_size+vocab_size+test_size)，将train_size中的后边部分（val_train_size）的单元标为True，剩下单元标为False
        test_mask, bool向量，长度为(train_size+vocab_size+test_size)，将test_size位置的单元标为True，剩下单元标为False
        train_size, 实数，训练集长度
        test_size，实数，测试集长度
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))  # labels = [train_y,vocab,test_y]
    # print(len(labels))

    train_idx_orig = parse_index_file("data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    '''
        以矩阵adj对角线为对称轴，可以将对称的两个元素组合为一个元组(Aij,Aji)，对该元组进行以较大者覆盖较小者的操作。
        例如一个对称元组(Aij,Aji)=(5,2)，则进行操作后的结果为(Aij,Aji)=(5,5)。
        对矩阵adj的全部对称元组进行上述操作，可以得到一个对称矩阵。
    '''
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    '''
        adj.T.multiply(adj.T > adj)：仅保留adj.T中大于adj对应位置的元素，不大于的元素变为0
    '''


    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def sparse_to_tuple(sparse_mx):
    """
        Convert sparse matrix to tuple representation.
        
        input:
            sparse_mx, 类型为sparse matirx
        return:
            返回该sparse_mx的三元组，(row,col),data,(矩阵行数，矩阵列数)
    """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo() # 将sparse matrix的格式转换为COO格式
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list): # 判断sparse_mx是否为类型list
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
        Row-normalize feature matrix and convert to tuple representation
        
        input:
            features, 矩阵
        return:
            对矩阵features进行行归一化，并返回对应的稀疏矩阵三元组，row, col, data
        功能：
            对矩阵features的每一行都进行归一化处理，使得每一行元素之和为1的矩阵

        例如输入的features为：
            array([[4, 0, 9, 0],
                   [0, 7, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 5]])
        输出的结果为：
            array([[0.30769231, 0.        , 0.69230769, 0.        ],
                   [0.        , 1.        , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 0.        ],
                   [0.        , 0.        , 0.        , 1.        ]])

    """
    rowsum = np.array(features.sum(1)) # 对于矩阵的每行求和，即 a*b的转化为 a*1
    r_inv = np.power(rowsum, -1).flatten() # 对a*1的矩阵的每个元素求倒数，然后flatten为一个list
    r_inv[np.isinf(r_inv)] = 0. # 将无穷小数转换为0
    r_mat_inv = sp.diags(r_inv) # 转化为对角矩阵
    features = r_mat_inv.dot(features) # 矩阵乘法
    return sparse_to_tuple(features) # 返回该稀疏矩阵的三元组，(row,col),data,(矩阵行数，矩阵列数)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^(-1/2), D is degree matrix

    '''
        adj是对称矩阵，因此 adj^T=adj.
        d_mat_inv_sqrt是对角矩阵，因此 d_mat_inv_sqrt^T=d_mat_inv_sqrt.
        (adj*d_mat_inv_sqrt)^T=(d_mat_inv_sqrt^T)*(adj^T) = d_mat_inv_sqrt*adj
        最后return: d_mat_inv_sqrt*adj*d_mat_inv_sqrt
    '''
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    '''
        原本的adj的对角线元素为空，这里对角线设置为1，即adj + sp.eye(adj.shape[0])
    '''
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()