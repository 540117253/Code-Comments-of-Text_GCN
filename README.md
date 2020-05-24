

## How to Use

1. Run `python remove_words.py 20ng`

2. Run `python build_graph.py 20ng`

3. Run `python train.py 20ng`

4. Change `20ng` in above 3 command lines to `R8`, `R52`, `ohsumed` and `mr` when producing results for other datasets.




- **本文主要内容：**
    - 本文主要是对以下论文，作者提供的代码进行注释解读：
《Liang Yao, Chengsheng Mao, Yuan Luo. "Graph Convolutional Networks for Text Classification." In 33rd AAAI Conference on Artificial Intelligence (AAAI-19), 7370-7377》

    - 论文作者的代码地址：https://github.com/yao8839836/text_gcn

    - 本文注释后的代码地址：https://github.com/540117253/Code-Comments-of-Text-GCN





- **任务描述：**

&emsp;&emsp;构建图卷积神经网络，对文本进行分类。

- **数据集文件：**
    -  'data/' + dataset + '.txt'： 每行代表一个数据样本，其包括文件路径、train-test标签、分类标签
    - 'data/corpus/' + dataset + '.txt'：每行对应一条样本的文本内容，每行的顺序与文件'data/' + dataset + '.txt'一致。




## 1. 数据预处理

#### 1.1 remove_words.py

**输入文件：**'data/corpus/' + dataset + '.txt'

**输出文件：**'data/corpus/' + dataset + '.clean.txt'

**处理过程：**
- 统计整个数据集的单词的词频，得到字典变量word_freq{}, key=单词, values=在数据集的出现次数
- 针对每条样本进行如下处理：将词频大于5且不在停用词库中的单词，拼接成一个string
- 将处理后的数据保存到'data/corpus/' + dataset + '.clean.txt'



#### 1.2 build_graph.py
- **读取文件'data/' + dataset + '.txt'，生成以下变量：**
    - doc_name_list, 列表，存放整个数据集样本（文本）的文件路径
    - doc_train_list，列表，存放训练集样本（文本）的文件路径
    - doc_test_list，列表，存放训练集样本（文本）的文件路径
- **读取文件'data/corpus/' + dataset + '.clean.txt'，生成以下变量：**
    - doc_content_list, 列表，存放整个数据集样本（文本）的内容
- **记录训练样本和测试样本的下标，生成以下变量：**
    - train_ids，列表，先存储doc_train_list中各条样本在doc_name_list的下标，然后进行打乱。变量train_ids被写入文件'data/' + dataset + '.train.index'
    - test_ids，列表，先存储doc_test_list中各条样本在doc_name_list的下标，然后进行打乱。变量test_ids被写入文件'data/' + dataset + '.train.index'
    - ids，列表，存储train_ids和test_ids，即id=train_ids+test_ids
- **根据train_ids和test_ids中打乱后的顺序，对doc_name_list和doc_content_list重新进行排列，分别得到以下变量：**
    - shuffle_doc_name_list，列表，根据train_ids和test_ids中打乱后的顺序，对doc_name_list重新排序。其存储的样本名称的顺序为先train_ids，后test_ids。shuffle_doc_name_list被写入文件'data/' + dataset + '_shuffle.txt'
    - shuffle_doc_words_list, 列表，根据train_ids和test_ids中打乱后的顺序，对doc_content_list重新排序。其存储的样本的内容的顺序为先train_ids，后test_ids。shuffle_doc_words_list被写入文件'data/corpus/' + dataset + '_shuffle.txt'
- **构建字典，并完成相关统计，得到如下变量：**
    - vocab，列表，存储整个数据集出现过的单词。vocab被写入文件'data/corpus/' + dataset + '_vocab.txt'
    - word_freq，字典，key=word，value=该单词在整个数据集中出现过的次数
    - word_doc_list，字典，key=word，value=整个数据集中包含该word的样本的id（该样本在shuffle_doc_words_list中的id）
    - word_doc_freq，字典，kye=word, value=整个数据集中包含该word的document（样本）数量
    - word_id_map，字典，key=word, value=该word在vocab中的id
- **记录label的的种类：**
    - label_list，集合set，整个数据集所出现的label集合。label_list被写入文件'data/corpus/' + dataset + '_labels.txt'
- **从train_ids选择90%作为真正的训练集，剩下的10%作为验证集：**
    - real_train_doc_names，列表，存储shuffle_doc_name_list的前'len(train_ids)*0.9'的单元。real_train_doc_names被写入文件'data/' + dataset + '.real_train.name'
    - real_train_size, 实数，real_train_size=len(train_ids)-int(0.1 * len(train_ids))
- **构建模型能够处理的数据：**
    - x，矩阵，大小为real_train_size*word_embeddings_dim，每行代表一个document的embedding。 如果不使用预训练的词向量，则该document的embedding为0。如果使用预训练的词向量，则该document的embedding为该document中各个单词embedding之和。
    - y，矩阵，大小为real_train_size*len(label_list)，每一行代表一个document对应label的one_hot向量
    - tx，矩阵，大小为test_size*word_embeddings_dim，每行代表一个document的embedding。 如果不使用预训练的词向量，则该document的embedding为0。如果使用预训练的词向量，则该document的embedding为该document中各个单词embedding之和。
    - ty，矩阵，大小为test_size*len(label_list)，每一行代表一个document对应label的one_hot向量
    - allx，矩阵, 大小为(train_size + vocab_size)*word_embeddings_dim，前train_size行存放document的embedding，后vocab_size行存放单词表各个单词的词向量。如果不使用预训练的词向量，词向量默认为随机初始化。
    - ally，矩阵，大小为(train_size + vocab_size)*len(label_list), 前train_size行存放document的label的one_hot编码，后vocab_size行都存放0向量
- **根据滑动窗口扫描的方式，统计以下变量：**
    - windows，列表，以大小为window_size（默认为10）、滑动步长为1的滑动窗口对整个数据集各个document的单词进行分组，分组结果存入列表windows中。例如windows[0]=['organization','university','maine'], windows[1]=['pin','map','din','cable']
    - word_window_freq，字典, key=单词，value=包含该单词的window的个数
    - word_pair_count，字典，key=单词1和单词2的id组合，value=包含该组合的window的数量（在window范围内，出现的共现次数）
- **计算邻接矩阵adj：**

&emsp; | 该部分的列号的范围是0 ~ train_size | 单词的id, 即word_id, 其范围是0~len(vocab)。该部分的列号的范围是(train_size+1) ~ (train_size+len(vocab)) | 该部分的列号的范围(doc_id+len(vocab)) ~ (len(shuffle_doc_name_list)+len(vocab)) 
---|---|---|---
训练样本的id，即doc_id，其范围是0 ~ train_size。该部分行号的范围是0 ~ train_size | null | word-doc的边权重tf_idf | null
单词的id，即word_id，其范围是0~len(vocab)。该部分的行号的范围是(train_size+1) ~ (train_size+len(vocab))  | null | word-word边的权重pmi | null
测试样本的id，即doc_id，其范围是(train_size+1) ~ len(shuffle_doc_name_list)。该部分行号的范围是(doc_id+len(vocab)) ~ (len(shuffle_doc_name_list)+len(vocab)) | null | word-doc的边权重tf_idf | null




## 2. 模型定义

&emsp;&emsp;该部分主要将模型分为两个文件进行编写: **models.py----模型结构、layers.py----模型层**
#### 2.1 models.py
&emsp;&emsp;参照keras的编写风格，采用类继承的方式来定义模型：首先定义一个抽象类Model，任何模型的实现都要基于父类Model进行继承来实现父类中的接口函数。
- **定义抽象类Model：**
```python
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self): # 当继承类没有实现函数def _build(self)而调用它，则会抛出错误
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name): 
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1]) # 将上一层的输出作为当前层的输入，计算当前层的输出hidden
            self.activations.append(hidden) # 保存当前层的输出
        self.outputs = self.activations[-1] # 该模型的输出为最后一层的输出

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
```

- **定义模型GCN：**

&emsp;&emsp;定义类GCN，要求其继承抽象类Model并实现接口函数def _loss(self)、def _accuracy(self)、def _build(self)、def predict(self)
```python
'''
两层的GCN，计算过程如下(实质就是两次矩阵乘法)：
    论文中的参数  代码中的参数   计算过程     维度
        X         features      null       n*n (n=train_size+vocab_size+test_size)
        A~         support      null       n*n
        W1         weight1      null       n*h1 (h1是模型参数)
        L1         output1     L1=A~XW1    n*h1 (L1是第一层GCN的输出)
        W2         weight2      null       h1*标签种数
        L2         output2     L2=A~L1W2   n*标签种数 (L2是第二层GCN的输出，每一行表示该样本预测的类别one-hot编码)
'''
class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build() # 调用抽象类Model的函数build()，触发自身实现的函数def _build(self)

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

    def _build(self): # 定义为两层相邻的GraphConvolution层

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            featureless=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x, #
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

```

#### 2.2 layers.py
&emsp;&emsp; 定义一个抽象类Layer，图卷积（GraphConvolution）层的实现要求继承类Layer并实现接口函数 def _call(self, inputs)，该函数主要用于被函数`def __call__(self, inputs)`调用。</br>
&emsp;&emsp; 其中函数`def __call__(self, inputs)`，使得该类或继承类的实例对象能够被直接调用，例子如下：
```python
class Entity:
'''调用实体来改变实体的位置。'''

def __init__(self, size, x, y):
    self.x, self.y = x, y
    self.size = size

def __call__(self, x, y):
    '''改变实体的位置'''
    self.x, self.y = x, y

e = Entity(1, 2, 3) // 创建实例
e(4, 5) //实例可以象函数那样执行，并传入x y值，修改对象的x y 
```
- **定义抽象类Layer**

```python
class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])
```
- **定义图卷积层GraphConvolution：**
```python
class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)): # 计算 A`XW
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)], sparse=self.sparse_inputs) # 计算XW
            else:
                pre_sup = self.vars['weights_' + str(i)] # 如果featureless==True, 则X为单位矩阵， XW直接等于W
            support = dot(self.support[i], pre_sup, sparse=True) # 计算 A`XW
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']
        self.embedding = output # the final output of this layer
        return self.act(output)
```




## 3. 训练模型(train.py)
- **加载预处理好的数据：**
    - adj，稀疏矩阵，原本预处理的非对称矩阵adj进行对称化后的矩阵，用于记录图中'word-word边'、'word-doc边'的权重。具体如何进行对称化，看（`utils.py中函数def load_corpus(dataset_str)`的注释）
    - features, 矩阵，大小为(train_size+vocab_size+test_size)*word_embeddings_dim，矩阵allx和矩阵tx进行纵向拼接后的矩阵
    - y_train, 矩阵，大小为(train_y+vocab+test_y)*len(label_list)，根据训练集样本的下标idx_train，在矩阵相应的行存放该样本的label的one-hot编码，其余行为0向量
    - y_val, 矩阵，大小为(train_y+vocab+test_y)*len(label_list)，根据验证集样本的下标idx_val，在矩阵相应的行存放该样本的label的one-hot编码，其余行为0向量
    - y_test, 矩阵，大小为(train_y+vocab+test_y)*len(label_list)，根据测试集样本的下标idx_test，在矩阵相应的行存放该样本的label的one-hot编码，其余行为0向量
    - train_mask, bool向量，长度为(train_size+vocab_size+test_size)，将train_size中的前边部分（real_train_size）的单元标为True，剩下单元标为False
    - val_mask, bool向量，长度为(train_size+vocab_size+test_size)，将train_size中的后边部分（val_train_size）的单元标为True，剩下单元标为False
    - test_mask, bool向量，长度为(train_size+vocab_size+test_size)，将test_size位置的单元标为True，剩下单元标为False
    - train_size, 实数，训练集长度
    - test_size，实数，测试集长度
    
- **训练模型：**
    - 使用训练集，正向传播，得到预测结果train_acc和损失函数train_loss，并反向传播进行参数更新
    - 使用验证集，正向传播，得到预测结果val_acc和损失函数val_loss
    - 如果当前的轮数t大于k（k为任意设定的值），且当前的val_loss大于之前第k到第t轮val_loss的平均值，则停止训练
    - 循环执行前3步
    
- **测试模型：**
    
    使用测试集，正向传播，得到预测结果
