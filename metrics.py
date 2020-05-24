import tensorflow as tf


'''
def masked_accuracy(preds, labels, mask):
    input:
        preds: 模型预测的结果。矩阵，大小为n*m, n是样本数，m是标签的种类数
        labels:模型预测的结果。矩阵，大小为n*m,  n是样本数，m是标签的种类数
        mask: bool向量，长度为n，用于标识现在参与计算准确率的样本。 由于该代码的特殊设置，放入模型进行推理的n个样本中，
              前train_size个样本作为训练，中间vocab个样本是词向量，最后test_size个样本作为测试。
              因此，n=(train_size+vocab+test_size)。
    output:
        实数，模型推理的准确率。
    例子：
            假设每个样本共有3种类别，放入模型进行推理的共有7个样本，其中前3个为训练样本（中间2个样本为词向量，后面2个样本为测
        试样本），现在需要计算这3个训练样本的推理准确率。
        input:
            preds:np.array([[3,1,0],
                            [0,1,2],
                            [0,2,9],
                            [4,2,0],
                            [1,2,5],
                            [0,3,1],
                            [0,0,5]
                            ])
            labels:np.array([[1,0,0],
                             [0,1,0],
                             [0,0,1],
                             [0,0,0],
                             [0,0,0],
                             [0,1,0],
                             [1,0,0]
                            ])
            mask:[True,True,True,False,False,False,False]
        计算过程：
            # 由于只有前3个样本参与准确率计算，因此准确率为 2/3。但存在mask的情况下，需要以下
            # 运算才能计算出 2/3.
            correct_prediction: [1,0,1,0,0,0,0]
            tf.reduce_mean(mask): 3/7
            mask /= tf.reduce_mean(mask): [7/3,7/3,7/3,0,0,0,0]
            accuracy_all *= mask: [7/3,0,7/3,0,0,0,0]
            tf.reduce_mean(accuracy_all): (7/3+7/3)/7 = 2/3
'''
def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1)) # correct_prediction 的内容如array([ True,  True, False], dtype=bool)

    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)




def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    print(preds)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)