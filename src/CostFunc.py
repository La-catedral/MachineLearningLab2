import numpy as np

def cost_func(X,W,T):
    size = T.shape[1]
    cost = 0
    for i in range(size):
        z = np.dot(W.T,X[:,i])
        cost = cost+(T[0,i]*z-np.log(1+np.exp(z)))/size
    # res_f = np.dot(np.dot(W.T,X),T.T)
    # res_l = np.sum(np.log(1+np.exp(np.dot(W.T,X))))
    # res = res_f - res_l
    # np.asscalar(res)
    return   cost#代价函数
