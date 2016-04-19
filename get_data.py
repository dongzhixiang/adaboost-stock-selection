#!/usr/bin/python
#coding=utf-8
__author__ = 'ZhaoYong'


import numpy as np
import pandas as pd

used_factors = ['PE', 'NetProfitGrowRate', 'MA10', 'MA60', 'LCAP', 'LFLO', 'NPToTOR', 'OperatingProfitGrowRate', 'TotalAssetGrowRate', 'DHILO', 'DEGM', 'Skewness', 'DAREC', 'GREC']
# 提取交易日
trading_date=DataAPI.TradeCalGet(exchangeCD="XSHG",beginDate="20100101",endDate="20160101",field=["calendarDate","isMonthEnd","isOpen"],pandas="1")

trading_date_open=trading_date[trading_date['isOpen'] == 1]
trading_date_open.to_csv('trading_date_open.csv')
# monthend day
monthend_day = trading_date_open[trading_date_open['isMonthEnd'] == 1]

for index in range(len(monthend_day['calendarDate'])):
    trading_day = monthend_day['calendarDate'].values[index]
    if trading_day >= '2011-01-01':
        monthend_train_day = monthend_day[monthend_day['calendarDate']<trading_day][-6:]

        for train_day in monthend_train_day['calendarDate']:
            
            universe_train = DataAPI.IdxConsGet(ticker="000300",intoDate=train_day,isNew="1",field=["consShortName","consID", "consTickerSymbol", "consExchangeCD"], pandas="1")
            universe_train.to_csv('universe_train'+str(train_day)+'.csv')
            stock=universe_train['consID'].values.tolist()
            stock_str_train = [str(x) for x in stock]
            factor =  DataAPI.MktStockFactorsOneDayGet(tradeDate=train_day,secID=stock_str_train,field=['secID','tradeDate']+used_factors,pandas="1")
            factor.to_csv('factor'+str(train_day)+'.csv')
            
        # 计算一个月后收益率
            
        universe_trading = DataAPI.IdxConsGet(ticker="000300",intoDate=trading_day,isNew="1",field=["consShortName","consID", "consTickerSymbol", "consExchangeCD"], pandas="1")
        universe_trading.to_csv('universe_trading'+str(trading_day)+'.csv')

        stock=universe_trading['consID'].values.tolist()
        stock_str_trading = [str(x) for x in stock]
        trading_day_ret = DataAPI.MktEqudGet(tradeDate=trading_day,secID=stock_str_trading,isOpen="1",field=["secID", 'preClosePrice'],pandas="1")
        trading_day_ret.to_csv('trading_day_ret.csv')
        train_day_price = DataAPI.MktEqudGet(tradeDate=trading_day,secID=stock_str_trading,isOpen="1",field=["secID", 'closePrice'],pandas="1")
        train_day_price.to_csv('train_day_price.csv')  
        
        old_trading_day=trading_date_open[trading_date_open['calendarDate'] < trading_day]['calendarDate'].values[-1]        
        
        universe_old_trading_day = DataAPI.IdxConsGet(ticker="000300",intoDate=old_trading_day,isNew="1",field=["consShortName","consID", "consTickerSymbol", "consExchangeCD"], pandas="1")
        universe_old_trading_day.to_csv('universe_old'+old_trading_day+'.csv')
        stock=universe_old_trading_day['consID'].values.tolist()
        stock_str = [str(x) for x in stock]
        factor_old_trading_day =  DataAPI.MktStockFactorsOneDayGet(tradeDate=old_trading_day,secID=stock_str,field=['secID','tradeDate']+used_factors,pandas="1")
        factor_old_trading_day.to_csv('factor_old'+old_trading_day+'.csv')    
        benchMark=DataAPI.MktIdxdGet(tradeDate=trading_day,ticker="000300",field='ticker,tradeDate,closeIndex')
        benchMark.to_csv('benchMark'+trading_day+'.csv')