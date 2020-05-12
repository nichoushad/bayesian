# bayesian
### 原理：

[贝叶斯原理](http://www.ruanyifeng.com/blog/2011/08/bayesian_inference_part_two.html)

[机器学习实战](https://github.com/Jack-Cherish/Machine-Learning)

贝叶斯处理垃圾邮件/可选择词袋、词集模式

[数据集在此下载]([https://plg.uwaterloo.ca/~gvcormac/treccorpus06/](https://links.jianshu.com/go?to=https%3A%2F%2Fplg.uwaterloo.ca%2F~gvcormac%2Ftreccorpus06%2F))

默认通过词集模型来进行表示。

词集模式：输入邮件的分词与词汇表对照,出现的标为1

词袋模式：输入邮件的分词与词汇表对照,出现的标为出现次数

对应函数

```python
def setOfWords2Vec(vocabList, inputSet):
    '''
    词集模型
    输入邮件的分词与词汇表对照,出现的标为1
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: returnVec: 文档向量，向量的每一元素为1或0
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        returnVec[vocabList.index(word)] = 1
    else:
        print('词: {word} 不在字典中'.format(word=word))
    return returnVec
```

```python
def bagOfWords2VecMN(vocabList, inputSet):
    '''
    词袋模型
    输入邮件的分词与词汇表对照,出现的标为出现次数
    :param vocabList: 词汇表
    :param inputSet: 某个文档
    :return: returnVec: 文档向量
    '''
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
```

## **1.数据集处理**

数据集中对data中的数据进行训练，而其训练参考结果在full中，full以例如<u>spam   path</u>的方式表示。

在get_index()函数中读取full中文件，代码如下

```python
index = pd.read_csv('./email/full/index', sep=' ', header=None, names=['type', 'path'], skiprows=skiprows, nrows=nrows)
```

参数skiprows指跳过几行开始读取，nrows指从该行开始读取几行，sep指读取中的分隔符，index文件中没有列标，所以给了个type，path的列标。

然后通过type和path组合，通过get_emailframe()函数获取相应path中的数据

## 2.处理数据

1.通过jieba分词，以及对非中文单词的过滤，得到词语列表，这里并没有使用过滤停用词，采用了把其中只有一个字的分词给省略。

2.创建字典，通过set()函数与并集来实现。

3.训练数据，得到词集

4.得到贝叶斯概率，假如邮件中有词集里没有的单词，就使用平滑处理

## 3.测试与结果分析

![img](file:///C:\Users\12273\AppData\Roaming\Tencent\Users\1227355064\TIM\WinTemp\RichOle\57~9LAKO5@`V7J4B2J4UQ5S.png)

其中

```python
accuracy = (TP + TN )/( TP + FP + TN + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
```

得到准确率、精确率和召回率。



个人理解：

准确率类似我们通常意义上的正确率，无论正负样本，都算进去

精确率相当于查准率，指看重是否有被误判为正样本的存在

召回率相当于正样本放出去收回的比例。



文件名a_b:a表示训练数量，b表示测试数量

|          | accuracy           | precision          | recall             |
| -------- | ------------------ | ------------------ | ------------------ |
| 500_200  | 0.81               | 1.0                | 0.7724550898203593 |
| 800_300  | 0.8266666666666667 | 0.9944444444444445 | 0.7782608695652173 |
| 1000_300 | 0.91               | 0.9797979797979798 | 0.8940092165898618 |
| 1500_400 | 0.9225             | 0.96875            | 0.9269102990033222 |
| 3000_500 | 0.952              | 0.9622093023255814 | 0.9678362573099415 |

500_200：{"accuracy": 0.81, "precision ": 1.0, "recall": 0.7724550898203593, "train_spam_in_rows": 430, "train_ham_in_rows": 70, "test_spam_in_rows": 167, "test_ham_in_rows": 33}

800_300:{"accuracy": 0.8266666666666667, "precision ": 0.9944444444444445, "recall": 0.7782608695652173, "train_spam_in_rows": 675, "train_ham_in_rows": 125, "test_spam_in_rows": 230, "test_ham_in_rows": 70}

1000_300:{"accuracy": 0.91, "precision ": 0.9797979797979798, "recall": 0.8940092165898618, "train_spam_in_rows": 827, "train_ham_in_rows": 173, "test_spam_in_rows": 217, "test_ham_in_rows": 83}

1500_400:{"accuracy": 0.9225, "precision ": 0.96875, "recall": 0.9269102990033222, "train_spam_in_rows": 1177, "train_ham_in_rows": 323, "test_spam_in_rows": 301, "test_ham_in_rows": 99}

3000_500:{"accuracy": 0.952, "precision ": 0.9622093023255814, "recall": 0.9678362573099415, "train_spam_in_rows": 2263, "train_ham_in_rows": 737, "test_spam_in_rows": 342, "test_ham_in_rows": 158}

可以看到，使用贝叶斯推断进行垃圾邮件的分类有非常良好的效果，而且准确率非常高。



