import numpy as np
import CostFunc

def logis_without_pun(X,W,T,num_item,alpha):
    #   W为n+1维包含w0的向量， X为n+1维、第一维为1的列向量组成的矩阵
    cost = []
    list = []
    for i in range(num_item):
        P_1 = 1 / (1 + np.exp(-1 * np.dot(W.T, X)))  # y=1的概率
        last_cost = -1 / T.shape[1] * CostFunc.cost_func(X, W, T)
        A = P_1 - T
        dW = np.dot(X,A.T)
        W = W - alpha * dW
        this_cost = -1/T.shape[1] * CostFunc.cost_func(X,W,T)
        # if last_cost - this_cost < 0.000001:
        #     break
        if last_cost < this_cost:
            alpha = alpha/2
        if i%10 == 0:
            cost.append( this_cost )
            list.append(i)
    return W,cost,list

def logis_with_pun(X,W,T,num_item,alpha,lambd):
    '''
    :param X: (m,n)
    :param W: (m,1)
    :param T: (1,n)
    :param num_item:
    :param alpha:
    :param lambd:
    :return:
    '''

    cost = []
    list = []
    for i in range(num_item):
        P_1 = np.exp(np.dot(W.T, X)) / (1 + np.exp(np.dot(W.T, X)))  # y=1的概率
        last_cost = -1 / T.shape[1] * CostFunc.cost_func(X, W, T)
        A =  T - P_1
        dW = 1/T.shape[1]*np.dot(X, A.T)
        W = W + alpha * dW - alpha * lambd * W #加入惩罚项
        this_cost = -1 / T.shape[1] * CostFunc.cost_func(X, W, T)
        if abs(this_cost)<0.00000001:
            break
        # if last_cost - this_cost < 0.00001:
        #     break
        if last_cost < this_cost:
            alpha = alpha/2
        if i % 10 == 0:
            cost.append(this_cost)
            list.append(i)
    return W, cost, list