import sklearn.model_selection
import numpy as np
import matplotlib.pyplot as plt
import gene_Samp
import logisticFunc

#对一个样本集计算预测准确率

def gene_W(dim):
    W = 0.1*np.random.rand(dim).reshape(-1,1)
    return W
def calc_Accu_with_logistic(X,W,T):
    Y = 1 / (1 + np.exp(-1*np.dot(W.T,X)))
    Y = (Y>=0.5).astype(int)
    res = Y^T #做异或运算 找出预测与原标记相同的样本
    return 1.0 - np.sum(res)/T.shape[1] #计算出比例

#画出训练得到的决策边界
def draw_decision_boundary(W):
    X1 = np.linspace(-1,1,100)
    X2 = (-1-W[1][0]*X1)/W[2][0]
    plt.plot(X1,X2,c = 'g')
    # plt.show()
    return X1,X2

def banknote_data_load():
    data = open('data_banknote_authentication.txt').readlines()
    data_set = []
    for data_line in data :
        this_line = data_line.strip().split(',')
        this_line_float = []
        for str in this_line:
            fl = float(str)
            this_line_float.append(fl)
        data_set.append(this_line_float)
    data_set = np.array(data_set)
    X = data_set[:, 0:4]
    Y_raw = data_set[:, 4]
    Y = []
    for i in Y_raw:
        Y.append(int(i))
    Y = np.array([Y]).T
    return X,Y

def seeds_data_load():
    data = open('seeds_dataset.txt').readlines()
    data_set = []
    for data_line in data :
        this_line = data_line.strip().split()
        this_line_float = []
        for str in this_line:
            fl = float(str)
            this_line_float.append(fl)
        if this_line_float[7] == 3:
            break
        data_set.append(this_line_float)
    data_set = np.array(data_set)
    X = data_set[:, 0:7]
    Y_raw = data_set[:, 7]
    Y = []
    for i in Y_raw:
        Y.append(int(i))
    Y = np.array([Y]).T
    return X,Y
#客户端




X,Y = banknote_data_load()
# X,Y = seeds_data_load()
# Y = Y -1

# X1, Y1,X2,Y2 = gene_Samp.gene_gauss_Samples(100, 100 ,True) #生成两类样本
# X1, Y1,X2,Y2 = gene_Samp.gene_gauss_Samples(100, 100 ,False) #生成两类样本


# X = np.c_[X1,X2].T #200,2 #合成总集合
# Y = np.c_[Y1,Y2].T #200,1

#将总集合随机划分为训练集和测试集
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.3)
#生成样本对应的特征矩阵 第一行为1

x_train = np.array(x_train).T #(m,n)
train_ones = np.array([np.ones(x_train.shape[1])])
x_train = np.r_[train_ones,x_train]
y_train = y_train.T #(1,n)
x_test = np.array(x_test).T
test_ones = np.array([np.ones(x_test.shape[1])])
x_test = np.r_[test_ones,x_test]
y_test = y_test.T


#随机生成函数
W = gene_W(X.shape[1]+1)
# W1,cost1,list1 = logisticFunc.logis_without_pun(x_train,W,y_train,1000,0.01)

W2,cost2,list2 = logisticFunc.logis_with_pun(x_train,W,y_train,1000,0.01,0.00001)
# print("the accuracy of the trainning set without punishment is :"+ str(calc_Accu_with_logistic(x_train,W1,y_train)))
# print("the accuracy of the test set without punishment is:"+str(calc_Accu_with_logistic(x_test,W1,y_test) ))
print("the accuracy of the trainning set with punishment is :"+ str(calc_Accu_with_logistic(x_train,W2,y_train)))
print("the accuracy of the test set with punishment is :"+ str(calc_Accu_with_logistic(x_test,W2,y_test)))

#画出决策边界
# plt.figure(1)
# draw_decision_boundary(W1)
# draw_decision_boundary(W2)
# plt.scatter(X1[0,:],X1[1,:],marker = 'o',label = 'class one') #第一类点
# plt.scatter(X2[0,:],X2[1,:],marker = 'x',label = 'class two') #第二类点
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('satisfy naive bayes,with pun,20 samples each')
# plt.legend()
#
plt.figure(2)
# plt.plot(list1,cost1,label = 'without')
plt.plot(list2,cost2 ,label = 'with')
plt.xlabel('epoch num ')
plt.ylabel('cost ')
# plt.title('cost with 10 samples each class')
plt.legend()
plt.show()