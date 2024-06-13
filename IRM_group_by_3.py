#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as scio 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# define function for evaluating portfolio's performance
def evaluate(data, freq=5):
    data = (1 + data).cumprod()#compute cumulative return
    
    # compute max drawdown
    def Max_DD(data):
        answer = 0
        for i in range(len(data)):
            if ((data.iloc[:i].max() - data.iloc[i]) / data.iloc[:i].max()) > answer:
                answer = ((data.iloc[:i].max() - data.iloc[i]) / data.iloc[:i].max())
            else:
                continue

        return answer * 100
    
    # compute annualized volatility
    def volatility(data):
        earning = data.pct_change().fillna(0)
        return earning.std() * np.sqrt(240/freq)

    # compute return
    def all_return(data):
        return (data.iloc[-1] - data.iloc[0]) / data.iloc[0]

    # compute annualized return
    def annual_return(data):
        return np.power((1 + all_return(data)), (240 / freq / len(data))) - 1

    # compute sharpe ratio
    def Sharpe(data,risk_free=0):
        return (annual_return(data) - risk_free) / volatility(data)    

    #compute the calmar ratio
    def Calmar(data):
        return annual_return(data)/Max_DD(data)*100
    
    # compute win ratio
    def Win_ratio(data):#P&L or return is positive means win
        new = data.pct_change().dropna()
        return len(new[new>0]) / len(new) * 100    
    
    def HWM(data):
        hwm = (data.max() - 1)*100
        return hwm
        
    # printing results
    for i in range(data.shape[1]):
        fct_data = data.iloc[:, i]
        print(str(fct_data.name))
        print('All_return: %.3f%%' % (all_return(fct_data) * 100))
        print('Annual_return: %.3f%%' % (annual_return(fct_data) * 100))
        print('Max_drawdown: %.2f%%' % Max_DD(fct_data))
        print('Sharpe_ratio: %.2f' % Sharpe(fct_data))
        print('Calmar_ratio: %.2f' % Calmar(fct_data))
        print('volatility: %.2f' % volatility(fct_data))
        print('Win_Ratio: %.2f%%' % Win_ratio(fct_data))
        print('HWM: %.2f%%' % HWM(fct_data))
        
        fct_data.plot(grid=True, figsize=(13, 8))
        plt.title("return")
        plt.show()


# In[3]:


def IC_RankIC(monthly_return, factor):
    '''
    Y: multi_inputs, dataframe with n obervations for m variables
    X: same size as Y(very important!!)
    '''
    monthly_return = monthly_return.shift(-1).dropna(how = 'all')
    factor = factor.loc[monthly_return.index]
    merged_df = pd.concat([monthly_return, factor],axis =1)

    # 初始化残差存储的DataFrame

    IC_df = pd.DataFrame(index=merged_df.index, columns=['IC','RankIC'])

    # 在每个时间点进行截面回归并计算残差
    for time_point, row in merged_df.iterrows():
        y = row.iloc[0:len(monthly_return.columns)].dropna()  # 因变量
        x = row.iloc[len(monthly_return.columns):].dropna()  # 自变量

        # 获取索引的交集
        columns = list(x.index.intersection(y.index))

        # 根据交集选择每个Series的值
        y = y.loc[columns]
        x = x.loc[columns]
#         if len(x) == 0:
#             print(time_point)
#         if len(y) == 0:
#             print(time_point)

        # 计算相关系数
        cov = (x-x.mean()).mul(y-y.mean()).sum()
        cov = cov/(len(x)-1)
        std = x.std()*y.std()
        IC = cov/std
        
        #计算相关系数
        rank_IC = x.corr(y,method = "spearman")

        # 将相关系数存储到IC_df中
        IC_df.loc[time_point, 'IC'] = IC
        IC_df.loc[time_point, 'RankIC'] = rank_IC
    # 打印残差结果
    return IC_df


# In[4]:


def get_res_Ngroups(res,n):
    columns = [f'G{i}' for i in range(n)]
    stock_group = pd.DataFrame(index = res.index, columns = columns)
    for time_point, row in res.iterrows():
        row = row.dropna()
        sorted_index = row.sort_values(ascending = True).index.to_list()
        num = np.floor(len(sorted_index)/n).astype(int)
        for i in range(n-1):
            start = i*num
            end = (i+1)*num
            stock_group.loc[time_point,f'G{i}'] = sorted_index[start:end]
        stock_group.loc[time_point, f'G{n-1}'] = sorted_index[(n-1)*num:]
    return stock_group


# In[5]:


def group_return(returns, group, group_num):
    #returns和group的freq需要相同
    group_return = pd.DataFrame(index= group.index, columns= group.columns)
    returns = returns.shift(-1)
    for time_point, groups in group.iterrows(): 
        for i in range(group_num):
            group_return.loc[time_point, f'G{i}'] = returns.loc[time_point][groups[i]].mean()        
    return group_return.shift(1)   


# In[6]:


data = pd.read_excel(r'C:\Users\1\Desktop\东吴金工\Q培训\行业日开盘收盘价格_处理后.xlsx', index_col = 0)
data


# In[7]:


#分组数量
n = 3


# In[8]:


# 选择列名中包含"open"的列
data_open = data.filter(regex='open')
data_open['date'] = data['date']
data_open = data_open.set_index('date')

# 使用列表推导式去掉每个字符串中的"open_"
columns_open = data_open.columns
open_columns = [column.replace("open_", "") for column in columns_open]
data_open.columns = open_columns
# 打印选择的列
data_open = data_open.loc['2018-05-01':,:]
data_open


# In[9]:


# 选择列名中包含"close"的列
data_close = data.filter(regex='close')
data_close['date'] = data['date']
data_close = data_close.set_index('date')

# 使用列表推导式去掉每个字符串中的"colse_"
columns_close = data_close.columns
close_columns = [column.replace("close_", "") for column in columns_close]
data_close.columns = close_columns

# 打印选择的列
data_close = data_close.loc['2018-05-01':,:]
data_close


# In[10]:


monthly_open = data_open.resample('M').first()
monthly_close = data_close.resample('M').last()
monthly_return = monthly_close.div(monthly_open).sub(1)
monthly_return = monthly_return.drop(index = '2018-05-31 00:00:00')
monthly_return


# ## 日内收益率

# In[11]:


intra_day = data_close.div(data_open).sub(1)
intraday_month = intra_day.rolling(20).mean().resample('M').last()
intraday_month = intraday_month.dropna(how = 'all')
intraday_month = intraday_month.drop(index = '2018-05-31 00:00:00')
intraday_month


# In[12]:


intraday_group = get_res_Ngroups(intraday_month, n)
intraday_group


# In[13]:


intraday_groupreturn = group_return(monthly_return, intraday_group, n)
intraday_groupreturn = intraday_groupreturn.fillna(0)
intraday_groupreturn ['hedge'] = - intraday_groupreturn.iloc[:,0] + intraday_groupreturn.iloc[:,n-1]
intraday_groupreturn


# In[14]:


intraday_groupreturn['market'] = intraday_groupreturn.mean(axis = 1)
intraday_cum_groupreturn = intraday_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
intraday_cum_groupreturn


# In[15]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 绘制DataFrame的多个列，并添加图例
#cum_group_return.iloc[:,:9].plot(figsize=(10, 6))
labels = [f'G{i}' for i in range(1,n+1)]
plt.plot(intraday_cum_groupreturn.iloc[:,:n],label=labels)
plt.plot(intraday_cum_groupreturn.iloc[:,n], color='red', linestyle='dashed', label='hedge')
# 设置横轴和纵轴标签
plt.xlabel('Date')
plt.ylabel('Cum return')
# 设置纵轴区间和刻度
#plt.ylim(0,100)
#plt.yticks([0, 5, 10, 15, 20, 25])
plt.legend(loc='best')
plt.figure(figsize = (10 ,6))

# 显示图形
plt.show()


# In[16]:


evaluate(intraday_groupreturn, freq = 20)


# ## 隔夜收益率

# In[17]:


over_night = data_open.div(data_close.shift(1)).sub(1)
overnight_month = over_night.rolling(20).mean().resample('M').last()
overnight_month = overnight_month.dropna(how = 'all')
overnight_month = overnight_month.drop(index = '2018-05-31 00:00:00')
overnight_month


# In[18]:


overnight_group = get_res_Ngroups(overnight_month, n)
overnight_group


# In[19]:


overnight_groupreturn = group_return(monthly_return, overnight_group, n)
overnight_groupreturn = overnight_groupreturn.fillna(0)
overnight_groupreturn ['hedge'] =  overnight_groupreturn.iloc[:,0] - overnight_groupreturn.iloc[:,n-1]
overnight_groupreturn


# In[20]:


overnight_groupreturn['market'] = overnight_groupreturn.mean(axis = 1)
overnight_cum_groupreturn = overnight_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
overnight_cum_groupreturn


# In[21]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 绘制DataFrame的多个列，并添加图例
#cum_group_return.iloc[:,:9].plot(figsize=(10, 6))
labels = [f'G{i}' for i in range(1,n+1)]
plt.plot(overnight_cum_groupreturn.iloc[:,:n],label=labels)
plt.plot(overnight_cum_groupreturn.iloc[:,n], color='red', linestyle='dashed', label='hedge')
# 设置横轴和纵轴标签
plt.xlabel('Date')
plt.ylabel('Cum return')
# 设置纵轴区间和刻度
#plt.ylim(0,100)
#plt.yticks([0, 5, 10, 15, 20, 25])
plt.legend(loc='best')
plt.figure(figsize = (10 ,6))

# 显示图形
plt.show()


# In[22]:


evaluate(overnight_groupreturn, freq = 20)


# ## 全天收益率

# In[23]:


all_day = data_close.pct_change()
allday_month = all_day.rolling(20).mean().resample('M').last()
allday_month = allday_month.dropna(how = 'all')
allday_month = allday_month.drop(index = '2018-05-31 00:00:00')
allday_month


# In[24]:


allday_group = get_res_Ngroups(allday_month, n)
allday_group


# In[25]:


allday_groupreturn = group_return(monthly_return, allday_group, n)
allday_groupreturn = allday_groupreturn.fillna(0)
allday_groupreturn ['hedge'] = - allday_groupreturn.iloc[:,0] + allday_groupreturn.iloc[:,n-1]
allday_groupreturn


# In[26]:


allday_groupreturn['market'] = allday_groupreturn.mean(axis = 1)
allday_cum_groupreturn = allday_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
allday_cum_groupreturn


# In[27]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 绘制DataFrame的多个列，并添加图例
#cum_group_return.iloc[:,:9].plot(figsize=(10, 6))
labels = [f'G{i}' for i in range(1,n+1)]
plt.plot(allday_cum_groupreturn.iloc[:,:n],label=labels)
plt.plot(allday_cum_groupreturn.iloc[:,n], color='red', linestyle='dashed', label='hedge')
# 设置横轴和纵轴标签
plt.xlabel('Date')
plt.ylabel('Cum return')
# 设置纵轴区间和刻度
#plt.ylim(0,100)
#plt.yticks([0, 5, 10, 15, 20, 25])
plt.legend(loc='best')
plt.figure(figsize = (10 ,6))

# 显示图形
plt.show()


# In[28]:


evaluate(allday_groupreturn.fillna(0), freq = 20)


# ## 隔夜-日内

# In[29]:


#标准化
def standardize(X):
    X = X.apply(lambda row: (row - row.mean())/row.std(), axis=1)
    return X

def winsor(X, p = 2):
    upper = np.percentile(X, 100 -p)
    lower = np.percentile(X, p)
    X[X>upper] = upper
    X[X<lower] = lower
    return X

def max_min(X):
    X = X.apply(lambda row: (row - row.min())/(row.max()-row.min()), axis=1)
    return X


# In[30]:


newF_minus = standardize(overnight_month).sub(standardize(intraday_month))
newF_minus


# In[31]:


newF_group = get_res_Ngroups(newF_minus, 3)
newF_group


# In[32]:


newFM_groupreturn = group_return(monthly_return, newF_group, 3)
newFM_groupreturn = newFM_groupreturn.fillna(0)
newFM_groupreturn ['hedge'] = newFM_groupreturn.iloc[:,0] - newFM_groupreturn.iloc[:,n-1]
newFM_groupreturn


# In[33]:


newFM_groupreturn['market'] = newFM_groupreturn.mean(axis = 1)
newFM_cum_groupreturn = newFM_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
newFM_cum_groupreturn


# In[34]:


evaluate(newFM_groupreturn, 20)


# In[61]:


IC_FM = IC_RankIC(monthly_return, newF_minus)
IC_FM.mean()


# In[62]:


IC_FM.mean()/IC_FM.std()*np.sqrt(12)


# ## 隔夜/日内

# In[35]:


newF_div = max_min(overnight_month).div(max_min(intraday_month))
newF_div


# In[64]:


newF_group = get_res_Ngroups(newF_div, 5)
newF_group


# In[65]:


newFD_groupreturn = group_return(monthly_return, newF_group, 5)
newFD_groupreturn = newFD_groupreturn.fillna(0)
newFD_groupreturn ['hedge'] = newFD_groupreturn.iloc[:,0] - newFD_groupreturn.iloc[:,n-1]
newFD_groupreturn


# In[66]:


newFD_groupreturn['market'] = newFD_groupreturn.mean(axis = 1)
newFD_cum_groupreturn = newFD_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
newFD_cum_groupreturn


# In[67]:


evaluate(newFD_groupreturn, 20)


# In[63]:


IC_FD = IC_RankIC(monthly_return, newF_div)
IC_FD.mean()


# # 隔夜-日内的净化

# In[40]:


import statsmodels.api as sm

def resi_cal(Y,X):
    '''
    Y: multi_inputs, dataframe with n obervations for m variables
    X: same size as Y(very important!!)
    '''
    
    merged_df = pd.concat([Y, X],axis =1)

    # 初始化残差存储的DataFrame

    residuals_df = pd.DataFrame(index=merged_df.index, columns=Y.columns)

    # 在每个时间点进行截面回归并计算残差
    for time_point, row in merged_df.iterrows():
        y = row.iloc[0:len(Y.columns)].dropna()  # 因变量
        x = row.iloc[len(Y.columns):].dropna()  # 自变量

        # 获取索引的交集
        columns = list(x.index.intersection(y.index))

        # 根据交集选择每个Series的值
        y = y.loc[columns]
        x = x.loc[columns]
        x = sm.add_constant(x)  # 添加常数项
        if len(x) == 0:
            print(time_point)
        if len(y) == 0:
            print(time_point)
        model = sm.OLS(y, x)  # 普通最小二乘回归模型
        results = model.fit()  # 拟合模型

        residuals = results.resid  # 获取残差
        residuals_df.loc[time_point,columns] = residuals

    # 打印残差结果
    return residuals_df


# In[41]:


res = resi_cal(newF_minus, allday_month)


# In[42]:


res


# In[43]:


n=5
new_group = get_res_Ngroups(res, n)
new_group


# In[44]:


new_groupreturn = group_return(monthly_return, new_group, n)
new_groupreturn = new_groupreturn.fillna(0)
new_groupreturn ['hedge'] = new_groupreturn.iloc[:,0] - new_groupreturn.iloc[:,n-1]
new_groupreturn


# In[45]:


new_groupreturn['market'] = new_groupreturn.mean(axis = 1)
new_cum_groupreturn = new_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
new_cum_groupreturn


# In[46]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# 绘制DataFrame的多个列，并添加图例
#cum_group_return.iloc[:,:9].plot(figsize=(10, 6))
labels = [f'G{i}' for i in range(1,n+1)]
plt.plot(new_cum_groupreturn.iloc[:,:n],label=labels)
plt.plot(new_cum_groupreturn.iloc[:,n], color='red', linestyle='dashed', label='hedge')
# 设置横轴和纵轴标签
plt.xlabel('Date')
plt.ylabel('Cum return')
# 设置纵轴区间和刻度
#plt.ylim(0,100)
#plt.yticks([0, 5, 10, 15, 20, 25])
plt.legend(loc='upper left')
plt.figure(figsize = (10 ,6))

# 显示图形
plt.show()


# In[47]:


evaluate(new_groupreturn, 20)


# In[58]:


IC_NewM = IC_RankIC(monthly_return, res)


# In[59]:


IC_df = IC_RankIC(monthly_return, res)
IC_df.mean()


# In[60]:


IC_df.mean()/IC_df.std()*np.sqrt(12)


# # 隔夜/日内的净化

# In[48]:


new_div = max_min(overnight_month).add(1).div(max_min(intraday_month).add(1))
new_div


# In[49]:


res1 = resi_cal(new_div, allday_month)
res1


# In[50]:


n = 4
newdiv_group = get_res_Ngroups(res1, n)
newdiv_group


# In[51]:


newdiv_groupreturn = group_return(monthly_return, newdiv_group, n)
newdiv_groupreturn = newdiv_groupreturn.fillna(0)
newdiv_groupreturn ['hedge'] = newdiv_groupreturn.iloc[:,0] - newdiv_groupreturn.iloc[:,n-1]
newdiv_groupreturn


# In[52]:


newdiv_groupreturn['market'] = newdiv_groupreturn.mean(axis = 1)
newdiv_cum_groupreturn = newdiv_groupreturn.fillna(0).apply(lambda x: x.add(1).cumprod())
newdiv_cum_groupreturn


# In[53]:


evaluate(newdiv_groupreturn, 20)


# In[ ]:





# In[54]:


# 将每行的残差数据分为十个组，每个组的数量相同
num_groups = 10+1
group_labels = range(1, num_groups + 1)
test = res.fillna(-99999999)
groups = test.apply(lambda x: pd.qcut(x, num_groups, labels=group_labels, duplicates='drop'), axis=1)
#将特定值的分组结果转换回缺失值
groups[groups == 11] = np.nan


# In[55]:


def get_res_10groups(res, group_num):
    stock_group = pd.DataFrame(index= res.index, columns= [])
    # 将每行的残差数据分为十个组，每个组的数量相同
    num_groups = group_num + 1
    group_labels = range(1, num_groups + 1)
    test = res.fillna(-99999999)
    groups = test.apply(lambda x: pd.qcut(x, num_groups, labels=group_labels, duplicates='drop'), axis=1)
    #将特定值的分组结果转换回缺失值
    groups[groups == num_groups] = np.nan
    for i in range(1,group_num+2):
        stock_group[f'G{i}'] = groups.apply(lambda x: x[x.eq(i)].index.tolist(), axis=1).values

    return stock_group.iloc[:,:-1]


# In[56]:


ggg = get_res_10groups(res, 10)


# In[57]:


ggg


# In[ ]:





# In[ ]:




