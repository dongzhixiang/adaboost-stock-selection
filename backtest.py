#!/usr/bin/python
#coding=utf-8
__author__ = 'ZhaoYong'

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
get_ipython().magic(u'matplotlib inline')
# 提取交易日

trading_date_open=pd.read_csv('data/trading_date_open.csv')

# monthend day
monthend_day = trading_date_open[trading_date_open['isMonthEnd'] == 1]
monthend_day=monthend_day[monthend_day['calendarDate'] >= '2011-01-01']
capital=1
ret=[]
bp=[]
for i in range(len(monthend_day['calendarDate'])-1):
    trading_day = monthend_day['calendarDate'].values[i]
    if trading_day >= '2011-01-01':  
        next_trading_day=monthend_day['calendarDate'].values[i+1]
        
        month_closePrice=pd.read_csv('data/buy_price_one'+trading_day+'.csv')
        next_month_closePrice=pd.read_csv('data/next_price_one'+next_trading_day+'.csv')
        month_data=pd.DataFrame(columns=['CurrentPrice','NextPrice','Return'])
        month_data['CurrentPrice']=month_closePrice['closePrice'].values
        month_data['NextPrice']=next_month_closePrice['closePrice'].values
        month_data['Return']=(month_data['NextPrice']-month_data['CurrentPrice'])/month_data['CurrentPrice']
        mean_ret=np.mean(month_data['Return'])
        capital=capital*(mean_ret+1)
        benchMark=pd.read_csv('data/benchMark'+trading_day+'.csv')        
        bmprice = benchMark['closeIndex'].values
        bp.append(bmprice)
        ret.append(capital)
rb = [bp[index][0]/bp[0][0] for index in range(len(bp))] 
ret_rb=pd.DataFrame(index=monthend_day['calendarDate'].values[:-1],columns=['ret','rb'])
ret_rb['ret']=ret
ret_rb['rb']=rb
ret_rb.plot(figsize=(16,8),grid=True)  # 净值曲线

# 年化收益率
year_ret=(ret_rb['ret'][-1]/ret_rb['ret'][0]-1)/5   
print year_ret

# 夏普比率
year_sigma=ret_rb['ret'].values.std()/5   #  年化收益标准差
# 无风险设为4%
sharpe=(year_ret-0.04)/year_sigma
print sharpe

# 最大回撤
down_list=[]
for i in range(1,len(ret_rb['ret'])):
    down=(1-ret_rb['ret'][i]/ret_rb['ret'][:i]).max()
    down_list.append(down)
print max(down_list)