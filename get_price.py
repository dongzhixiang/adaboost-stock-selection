#!/usr/bin/python
#coding=utf-8
__author__ = 'ZhaoYong'


import numpy as np
import pandas as pd
import adaboost



used_factors = ['PE', 'NetProfitGrowRate', 'MA10', 'MA60', 'LCAP', 'LFLO', 'NPToTOR', 
                'OperatingProfitGrowRate', 'TotalAssetGrowRate', 'DHILO', 'DEGM', 'Skewness', 
                'DAREC', 'GREC']

# 提取交易日

trading_date_open=pd.read_csv('data/trading_date_open.csv')

# monthend day
monthend_day = trading_date_open[trading_date_open['isMonthEnd'] == 1]
buy_list_all=[]
for index in range(len(monthend_day['calendarDate'])):
    trading_day = monthend_day['calendarDate'].values[index]
    if trading_day >= '2011-01-01':
        monthend_train_day = monthend_day[monthend_day['calendarDate']<trading_day][-12:]
    # 沪深300成分股

        features=[]
        labels=[]
        for index in range(len(monthend_train_day['calendarDate'])):
            train_day = monthend_train_day['calendarDate'].values[index]

            universe_train = pd.read_csv('data/universe_train'+train_day+'.csv')

            factor = pd.read_csv('data/factor'+train_day+'.csv')

            factor['ticker'] = factor['secID'].apply(lambda x: x[0:6])
            factor.set_index('ticker',inplace=True)


            for f in used_factors:
                if factor[f].std() == 0:
                    continue
                factor[f] = (factor[f] - factor[f].mean()) / factor[f].std()

            factor=factor.dropna()

            # rank data
            factor.iloc[:,3:]=factor.iloc[:,3:].rank(method='first').apply(lambda x : x/len(factor))
            universe_trading = pd.read_csv('data/universe_trading'+trading_day+'.csv')
            stock=universe_trading['consID'].values.tolist()
            stock_str_trading = [str(x) for x in stock]
            trading_day_ret=pd.read_csv('data/trading_day_ret.csv')
            train_day_price=pd.read_csv('data/train_day_price.csv')
            trading_day_ret['train_ret'] = trading_day_ret['preClosePrice']/train_day_price['closePrice']-1
            # 读进来数据，多了一列，删除生成新的
            factor_new=factor.drop('Unnamed: 0',axis=1)
            trading_day_ret_new=trading_day_ret.drop('Unnamed: 0', axis=1)
            train_set = pd.merge(factor_new, trading_day_ret_new, on='secID')
            train_set['ticker'] = train_set['secID'].apply(lambda x: x[0:6])
            train_set.set_index('ticker',inplace=True)
            train_set.dropna()
            train_set=train_set.sort(columns='train_ret', ascending=False)
            # 取前后百分之三十，其他舍去
            label_1=train_set.ix[:int(0.3*len(train_set)),:]
            label_0=train_set.ix[-int(0.3*len(train_set)):,:]
            label_1['train_ret']=1
            label_0['train_ret']=0
            feature_one=pd.concat([label_1,label_0]).iloc[:,2:-2].values.tolist()
            for i in range(len(feature_one)):
                features.append(feature_one[i])
            label_one=pd.concat([label_1,label_0])['train_ret'].values.tolist()
            for i in range(len(label_one)):
                labels.append(label_one[i])



        # train classifier
        
        classifierArr = adaboost.adaBoostTrainDS(mat(features), labels, 30)

        old_trading_day=trading_date_open[trading_date_open['calendarDate'] < trading_day]['calendarDate'].values[-1]        

        predict_data = pd.read_csv('data/factor_old'+old_trading_day+'.csv')
        predict_data=predict_data.dropna()
        predict_data.iloc[:,3:]=predict_data.iloc[:,3:].rank(method='first').apply(lambda x : x/len(predict_data))
        x=predict_data.iloc[:,3:].values.tolist()
        
        # predict
        y=adaboost.adaClassify(x,classifierArr)
        predict_label=predict_data.loc[:,['secID','tradeDate']]
        predict_label['pro']=y
        predict_label['label']=sign(y)
        buy=predict_label[predict_label['label'] == 1]
        buy=buy.sort(columns=['pro'],ascending=False)[:45]
        buy_list_one=buy['secID'].values.tolist()
        buy_list_all.append(buy_list_one)
        
        
        for i in range(len(buy_list_all)):
            buylist_one=buy_list_all[i]
            if trading_day < '2015-12-31':
                next_trading_day = monthend_day[monthend_day['calendarDate']>trading_day]['calendarDate'].values[0]
                buy_price_one = DataAPI.MktEqudAdjGet(tradeDate=trading_day,secID=buylist_one,field=["tradeDate", "secID", "ticker", "closePrice","isOpen"],pandas="1") 
                
                next_price_one = DataAPI.MktEqudAdjGet(tradeDate=next_trading_day,secID=buylist_one,field=["tradeDate","secID", "ticker", "closePrice", "isOpen"],pandas="1")
                
                buy_price_one.to_csv('buy_price_one'+trading_day+'.csv')
                next_price_one.to_csv('next_price_one'+next_trading_day+'.csv')

