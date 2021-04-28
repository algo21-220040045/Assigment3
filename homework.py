# -*- coding: utf-8 -*-
"""
Created on Thu April 27 15:23:27 2021

@author: Steven_Xin
"""
#aim：反算基金仓位
#tools：ols、wls、pca、step_wise、ridge、lasso
#要算的东西：横截面预测值减去实际值的均值和方差

import numpy as np
import pandas as pd
import datetime
from sklearn import linear_model
from sklearn.decomposition import PCA
from cvxopt import solvers as so
from cvxopt import matrix as ma
#cvxopt包安装教程 https://blog.csdn.net/qq_32106517/article/details/78746517

file_data = './Data.xlsx' #-- 数据文件所在路径
len_window = 50 #-- 回归时间窗口
bound = [0.6,1] #-- 仓位下上限，普通股票型基金下限为0.8，偏股型基金下限为0.6
lasso_lambda = 0.002 #-- Lasso回归参数：正则化项惩罚系数
ridge_lambda = 0.002
    
#-- 加权最小二乘法权重，主函数里回归时用
wgt = np.sqrt(np.linspace(1,len_window,len_window)/len_window)
#------------------------------------------#
#输入数据
df_fund_nav = pd.read_excel(file_data,sheet_name='fund') 
#基金价格 （2016，09，30）-（2019,01,25）
df_fund_ret = df_fund_nav / df_fund_nav.shift(1) - 1 
#基金日收益率（2016，9，30）-（2019,01,25）
df_indus_nav = pd.read_excel(file_data,sheet_name='csindus')
#指数价格 （2016，09，30）-（2019,01,25）
df_indus_ret = df_indus_nav / df_indus_nav.shift(1) - 1
#指数日收益率（2016，9，30）-（2019,01,25）
df_fund_ret, df_indus_ret = df_fund_ret.iloc[1:,:],  df_indus_ret.iloc[1:,:]
# 基金和指数的日收益率 （2016，10，10）-（2019,01,25）
df_fund_pos = pd.read_excel(file_data,sheet_name='position')/100
#基金仓位（2016，9，30）-（2019,01,25）


#code_fund='000020.OF'
#date_start=(2019,1,25)
#date_end=(2019,1,25)

def func_lin(df_X,sr_y):
    reg = linear_model.LinearRegression()
    reg.fit(df_X, sr_y)
    return reg.coef_, reg.intercept_
#coef_lin,intercept_lin = func_lin(df_X,sr_y)  
#tmp_total = coef_lin.sum()  

def func_pca(df_X,sr_y):
    pca = PCA(n_components=0.95)
    pca.fit(df_X)
    df_X_pca = pca.transform(df_X)
    eigval, eigvec = np.linalg.eig(np.cov(np.matrix(df_X).I))
    return func_lin(df_X_pca*np.matrix(eigvec[:,0:df_X_pca.shape[1]]).I,sr_y) 

#pca = PCA(n_components=0.95)
#pca.fit(df_X)
#df_X_pca = pca.transform(df_X)
#df_X_pca.shape[1]
#eigval, eigvec = np.linalg.eig(np.cov(np.matrix(df_X).I))
#df_X_pca*np.matrix(eigvec[:,0:df_X_pca.shape[1]]).I


#coef_lin,intercept_lin = func_pca(df_X,sr_y)  
#tmp_total = coef_lin.sum()  
#print('各主成分贡献度:{}'.format(pca.explained_variance_ratio_))
#pca.singular_values_ 
#import statsmodels.api as sm
#ols = sm.OLS(sr_y,df_X).fit()
#ols.summary()

def func_ridge(df_X,sr_y,ridge_lambda):
    ridge_alpha = ridge_lambda/df_X.shape[0]/2
    reg = linear_model.Ridge(alpha=ridge_alpha,fit_intercept=True,normalize=False,tol=1e-4)
    reg.fit(df_X, sr_y)
    return reg.coef_, reg.intercept_

#ridge_lambda = 0.002
#coef_ridge,intercept_ridge = func_ridge(df_X,sr_y,ridge_lambda)  
#tmp_total = coef_ridge.sum()  

#-- lasso回归函数
def func_lasso(df_X,sr_y,lasso_lambda):
    lasso_alpha = lasso_lambda/df_X.shape[0]/2 #-- Lasso回归参数
    reg = linear_model.Lasso(alpha=lasso_alpha,fit_intercept=True,normalize=True,tol=1e-4,positive=True)
    reg.fit(df_X,sr_y)
    return reg.coef_, reg.intercept_
#------------------------------------------#

#-- 二次优化函数
def func_qp(X,y,lb,ub):
    # X为指数日收益率矩阵，维度为90×29； y为基金收益率序列，维度为90×1 ；
    #lb，ub分别为下限和上限
    X,y = np.array(X),np.array(y)
    n_bases = X.shape[1]
    #-- 使用cvopt凸优化包中的二次优化, 对回归的平方误差进行优化
    #-- 二次项矩阵P=X'*X, dim=N*N; 一次项矩阵q, dim=N*1
    #-- 优化1/2*w'Pw+q'w
    #-- 同时加入限制条件Gw<=h, Aw=b, dimG=j*N, dimh=j*1, dimA=k*N, dimb=k*1
    #-- j为不等式约束数目, k为等式约束数目
    P = np.dot(np.transpose(X),X)
    q = -1 * np.dot(np.transpose(X),y)
    
    G = np.concatenate((np.diag(np.ones(n_bases)),\
                        -1*np.diag(np.ones(n_bases)),\
                        np.ones([1,n_bases]),\
                        -1*np.ones([1,n_bases])),axis=0)
    h = np.concatenate((ub*np.ones([n_bases,1]),\
                        np.zeros([n_bases,1]),\
                        np.array([[ub],[-lb]])),axis=0)
    #-- 使用cvxopt进行优化时需要转化为cvxopt库内matrix格式
    X,y,P,q,G,h = ma(X),ma(y),ma(P),ma(q),ma(G),ma(h)
    #-- 核心部分
    sol = so.qp(P, q, G, h)                     
    w = np.array(sol['x'])
    return w
    # w是29个指数的回归系数序列，对其求和的结果可作为仓位的估计值

#coef_qp = func_qp(df_X,sr_y,bound[0],bound[1])
#tmp_total = coef_qp.sum()  

def cal(code_fund,date_start,date_end,method):
    #输入基金代码，可输出仓位预测值，计算某一天的或者时间序列的均可
    #------------------------------------------#
    #-- 确定仓位测算起止日期
    date_start = datetime.datetime(date_start[0],date_start[1],date_start[2])
    date_end = datetime.datetime(date_end[0],date_end[1],date_end[2])
    if date_start > date_end:
        raise ValueError("date_start must be prior to date_end.")
    if not date_start in df_fund_ret.index:
        raise ValueError("date_start must be a trading day.")
    if not date_end in df_fund_ret.index:
        raise ValueError("date_end must be a trading day.")
    i_date_start = np.where(df_fund_ret.index==date_start)[0][0] #开始日期在基金日收益率数据集的哪一行
    i_date_end = np.where(df_fund_ret.index==date_end)[0][0]#结束日期在基金日收益率数据集的哪一行
    i_fund = np.where(df_fund_ret.columns==code_fund)[0][0]#选取基金在基金日收益率数据集的哪一列

    saved_results = pd.DataFrame(index=df_fund_ret.index[i_date_start:i_date_end+1],columns=[code_fund])
    
    
    for i_day in range(i_date_start,i_date_end+1):
    #-- 指数收益乘以权重作为自变量
        df_X = df_indus_ret.iloc[i_day-len_window+1:i_day+1,:]
        df_X = np.multiply(df_X,np.tile(wgt,(df_X.shape[1],1)).T)
        sr_y = df_fund_ret.loc[df_X.index,df_fund_ret.columns[i_fund]].copy()
        
        if np.isnan(sr_y).sum() == 0:
            sr_y.rename('fund_ret',inplace=True)
            sr_y = np.multiply(sr_y,wgt)
            
            
            if method=='lasso':
               coef,intercept = func_lasso(df_X,sr_y,lasso_lambda)
            elif method=='linear':
               coef,intercept = func_lin(df_X,sr_y)
            elif method=='ridge':
               coef,intercept = func_ridge(df_X,sr_y,ridge_lambda)
            elif method=='pca':
               coef,intercept = func_pca(df_X,sr_y) 
            elif method=='qp':
               coef = func_qp(df_X,sr_y,bound[0],bound[1])
            else:
               print("请输入'lasso','linear','ridge','pca','qp'中的一个。") 

            tmp_total = coef.sum()
            if tmp_total < bound[0]:
                tmp_total = bound[0]
            elif tmp_total > bound[1]:
                tmp_total = bound[1]
            saved_results.loc[df_X.index[-1],[code_fund]] = tmp_total
                
    return saved_results

def cal_cro1(date,method):
    #输入横截面日期，输出日期的预测平均仓位，真实平均仓位，两者差的平均值和方差
    datein = datetime.datetime(date[0],date[1],date[2])
    i_datein = np.where(df_fund_pos.index==datein)[0][0]
    pos_re=df_fund_pos.iloc[i_datein,:]
    pos_em=[0]*476
    diff=[0]*476
    for i in range(476):
        pos_em[i]=cal(df_fund_nav.columns[i],date,date,method).iloc[0,0]
        diff[i]=pos_em[i]-pos_re[i]
        
    return np.mean(pos_em),np.mean(pos_re),np.mean(diff),np.std(diff)        
    
a,b,c,d=cal_cro1((2017,12,29),'lasso')

# def cal_num(diff):
#     d1,d2,d3,d4,d5=0,0,0,0,0
#     for i in range(len(diff)):
#         if diff[i]<=-0.2:
#            d1+=1
#         elif diff[i]<=-0.1:
#            d2+=1
#         elif diff[i]<=0:
#            d3+=1
#         elif diff[i]<=0.2:
#            d4+=1
#         else:
#            d5+=1
#     return [d1,d2,d3,d4,d5]


#算超参数的
daylen=[]
for i in np.linspace(0.2,0.1,num=10):
    lasso_lambda=i
    daylen.append(cal_cro1((2018,6,29),'lasso'))

for i in np.linspace(0.02,0.01,num=10):
    lasso_lambda=i
    daylen.append(cal_cro1((2018,3,30),'lasso'))

for i in np.linspace(0.002,0.001,num=10):
    lasso_lambda=i
    daylen.append(cal_cro1((2018,3,30),'lasso'))
for i in np.linspace(0.0002,0.0001,num=10):
    lasso_lambda=i
    daylen.append(cal_cro1((2018,3,30),'lasso'))
daylen1=[]
for i in np.linspace(0.2,0.1,num=10):
    ridge_lambda=i
    daylen1.append(cal_cro1((2018,3,30),'ridge'))
for i in np.linspace(0.02,0.01,num=10):
    ridge_lambda=i
    daylen1.append(cal_cro1((2018,3,30),'ridge'))
for i in np.linspace(0.002,0.001,num=10):
    ridge_lambda=i
    daylen1.append(cal_cro1((2018,3,30),'ridge'))
for i in np.linspace(0.0002,0.0001,num=10):
    ridge_lambda=i
    daylen1.append(cal_cro1((2018,3,30),'ridge'))
    


#def detail(code_fund,date_start,date_end,method):
#    date_start = datetime.datetime(date_start[0],date_start[1],date_start[2])
#    date_end = datetime.datetime(date_end[0],date_end[1],date_end[2])
#    if date_start > date_end:
#        raise ValueError("date_start must be prior to date_end.")
#    if not date_start in df_fund_ret.index:
#        raise ValueError("date_start must be a trading day.")
#    if not date_end in df_fund_ret.index:
#        raise ValueError("date_end must be a trading day.")
#    i_date_start = np.where(df_fund_ret.index==date_start)[0][0] #开始日期在基金日收益率数据集的哪一行
#    i_date_end = np.where(df_fund_ret.index==date_end)[0][0]#结束日期在基金日收益率数据集的哪一行
#    i_fund = np.where(df_fund_ret.columns==code_fund)[0][0]#选取基金在基金日收益率数据集的哪一列
#    
#    for i_day in range(i_date_start,i_date_end+1):
#    #-- 指数收益乘以权重作为自变量
#        df_X = df_indus_ret.iloc[i_day-len_window+1:i_day+1,:]
#        sr_y = df_fund_ret.loc[df_X.index,df_fund_ret.columns[i_fund]].copy()
#        if method=='Lasso':
#            coef,intercept = func_lasso(df_X,sr_y,lasso_lambda)
#        elif method=='OLS':
#            coef,intercept = func_lin(df_X,sr_y)
#        elif method=='Ridge':
#            coef,intercept = func_ridge(df_X,sr_y,ridge_lambda)
#        elif method=='PCR':
#            coef,intercept = func_pca(df_X,sr_y) 
#        elif method=='qp':
#            coef = func_qp(df_X,sr_y,bound[0],bound[1])
#        else:
#            print("请输入'Lasso','Linear','Ridge','PCR','qp'中的一个。") 
#    return coef * 100
#
#detail('000020.OF',(2018,5,10),(2018,6,21),"qp")
#detail('000020.OF',(2018,5,10),(2018,6,21),"OLS")
#np.round(detail('000020.OF',(2018,5,10),(2018,6,21),"qp"),decimals=8)


#sample    
cal('000020.OF',(2018,1,10),(2019,1,10),'qp')     


