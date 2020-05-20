import pandas as pd
import re
import jieba
import json
import os

from numpy import *
from sklearn.utils import shuffle
def set_result_to_file(result):
    train_num = result['train_spam_in_rows']+result['train_ham_in_rows']
    test_num = result['test_spam_in_rows']+result['test_ham_in_rows']
    result_path = os.path.join('./result','{train}_{test}'.format(train=train_num,test=test_num))
    with open(result_path, 'a') as f:
        f.write('\n')
        json.dump(result, f)


def clean_str(string):
    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def get_index(skiprows,nrows):
    index = pd.read_csv('./email/full/index', sep=' ', header=None, names=['type', 'path'], skiprows=skiprows, nrows=nrows)
    index = shuffle(index)
    print("data shape:", index.shape)
    spams_in_rows = index.loc[index['type'] == "spam"].shape[0]
    ham_in_rows = index.loc[index['type'] == "ham"].shape[0]
    spam_path = index[index["type"] == "spam"]["path"]
    ham_path = index[index["type"] == "ham"]["path"]
    path = index["path"]
    return index


def createVocabList(dataSet):
    '''
    创建一个包含在所有文档中出现的不重复词的列表
    :param dataSet: 词
    :return: vocabset: 词汇表
    '''
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 并集
    return list(vocabSet)




# 通过index文件获取spam和ham对应的文件的path,并且处理数据
def get_emailframe(path):
    with open(os.path.join('.', 'email', 'data', path), 'r', encoding='gbk', errors='ignore') as f:
        lines = f.readlines()
        email = ''
        for line in lines:
            line = clean_str(line)
            email += line
        f.close()
        email_word = [word for word in jieba.cut(email) if word.strip() != '']
        return email_word



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


def trainNB0(trainMatrix, trainCategory):
    '''

    :param trainMatrix:
    :param trainCategory:
    :return:

    '''
    # numtraindocs 邮件总数
    numTrainDocs = len(trainMatrix)
    # 词袋中词汇总数
    numWords = len(trainMatrix[0])
    # pSpam 垃圾邮件百分比
    pSpam = sum(trainCategory) / float(numTrainDocs)
    #拉普拉斯平滑 由于2分类 每个词汇数量+1，词汇总数+2 防止0/0
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+= trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    # 某词语在所有词语中的比例 加上log防止过小
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect,p1Vect,pSpam

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''

    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1>p0:
        return 1
    else:
        return 0


def spamTest(trainNum, testNum):
    docList = []
    classList = []
    index = get_index(0,trainNum+testNum)
    spams_in_rows = index.loc[index['type'] == "spam"].shape[0]
    ham_in_rows = index.loc[index['type'] == "ham"].shape[0]
    print(spams_in_rows,ham_in_rows)
    for type, path in zip(index['type'], index['path']):
        wordList = get_emailframe(path)
        docList.append(wordList)  # 用来创建字典
        if type == 'spam':
            classList.append(1)  # 1代表垃圾邮件
        else:
            classList.append(0)
    vocabList = createVocabList(docList)  # 创建词典
    trainingSet = range(trainNum)
    testSet = range(trainNum, trainNum + testNum)
    trainMat = [] # 训练向量
    trainClasses = [] # 训练类
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # accuracy=0
    # precision=0
    # recall=0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            if(classifyNB(array(wordVector), p0V, p1V, pSpam)):
                FP+=1
                print("分类错误", docList[docIndex],'\n',classList[docIndex])
            else:
                FN+=1
        else:
            if (classifyNB(array(wordVector), p0V, p1V, pSpam)):
                TP+=1
            else:
                TN+=1
    accuracy = (TP + TN )/( TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    train_type = get_index(0,trainNum)
    test_type = get_index(trainNum,testNum)

    train_spam_in_rows = train_type.loc[train_type['type'] == "spam"].shape[0]
    train_ham_in_rows = train_type.loc[train_type['type'] == "ham"].shape[0]
    test_spam_in_rows = test_type.loc[test_type['type'] == "spam"].shape[0]
    test_ham_in_rows = test_type.loc[test_type['type'] == "ham"].shape[0]
    result = {}
    result['accuracy']=accuracy
    result['precision'] = precision
    result['recall'] = recall
    result['train_spam_in_rows'] = train_spam_in_rows
    result['train_ham_in_rows'] = train_ham_in_rows
    result['test_spam_in_rows'] = test_spam_in_rows
    result['test_ham_in_rows'] = test_ham_in_rows
    set_result_to_file(result)
    print(result)

    # return vocabList,fullText


if __name__ == '__main__':
    # spamTest(800,300)
    # spamTest(800, 300)
    # spamTest(800, 300)
    # spamTest(800, 300)
    # spamTest(1000,300)
    # spamTest(1000, 300)
    # spamTest(1000, 300)
    # spamTest(1500,400)
    # spamTest(1500, 400)
    # spamTest(1500, 400)
    spamTest(3000,500)
    spamTest(3000, 500)

#统计正确率（accuracy）、准确率(Precision)、召回率（recall）；