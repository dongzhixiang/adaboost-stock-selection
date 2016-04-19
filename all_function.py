
# coding: utf-8

# In[ ]:

from numpy import *
import pandas as pd

                           
used_factors = ['PE', 'NetProfitGrowRate', 'MA10', 'MA60', 'LCAP', 'LFLO', 'NPToTOR', 'OperatingProfitGrowRate', 'TotalAssetGrowRate', 'DHILO', 'DEGM', 'Skewness', 'DAREC', 'GREC']
def get_stock(day):
    universe = DataAPI.IdxConsGet(ticker="000300",intoDate=day,isNew="1",field=["consShortName","consID", "consTickerSymbol", "consExchangeCD"], pandas="1")
    stock=universe['consID'].values.tolist()
    stock_str = [str(x) for x in stock]
    return stock_str
def get_factor(day):
    stock_str=get_stock(day)
    factor =  DataAPI.MktStockFactorsOneDayGet(tradeDate=day,secID=stock_str,field=['secID','tradeDate']+used_factors,pandas="1")
    factor['ticker'] = factor['secID'].apply(lambda x: x[0:6])
    factor.set_index('ticker',inplace=True)
    for f in used_factors:
        if factor[f].std() == 0:
            continue
        factor[f] = (factor[f] - factor[f].mean()) / factor[f].std()
    factor=factor.dropna() 
    # rank data
    factor.iloc[:,2:]=factor.iloc[:,2:].rank(method='first').apply(lambda x : x/len(factor))
    return factor
# 提取交易日
trading_date=DataAPI.TradeCalGet(exchangeCD="XSHG",beginDate="20100101",endDate="20160101",field=["calendarDate","isMonthEnd","isOpen"],pandas="1")
trading_date_open=trading_date[trading_date['isOpen'] == 1]
# monthend day
monthend_day = trading_date_open[trading_date_open['isMonthEnd'] == 1]

capital=1
ret=[]
buy_list_all=[]

for index in range(len(monthend_day['calendarDate'])):
    trading_day = monthend_day['calendarDate'].values[index]
    if trading_day >= '2011-01-01':
        monthend_train_day = monthend_day[monthend_day['calendarDate']<trading_day][-6:]
    # 沪深300成分股
        features=[]
        labels=[]
        for train_day in monthend_train_day['calendarDate']:
            train_data=get_factor(train_day)
            # 计算一个月后收益率
            stock_str=get_stock(trading_day)
            trading_day_ret = DataAPI.MktEqudGet(tradeDate=trading_day,secID=stock_str,isOpen="1",field=["secID", 'preClosePrice'],pandas="1")
            train_day_price = DataAPI.MktEqudGet(tradeDate=train_day,secID=stock_str,isOpen="1",field=["secID", 'closePrice'],pandas="1")
            trading_day_ret['train_ret'] = trading_day_ret['preClosePrice']/train_day_price['closePrice']-1
           
            train_set = pd.merge(train_data, trading_day_ret, on='secID')
            train_set['ticker'] = train_set['secID'].apply(lambda x: x[0:6])
            train_set.set_index('ticker',inplace=True)
            train_set.dropna()
            train_set=train_set.sort(columns='train_ret', ascending=False)
            # 取前后百分之三十，其他舍去
            label_1=train_set.ix[:int(0.3*len(train_set)),:]
            label_0=train_set.ix[-int(0.3*len(train_set)):,:]
            label_1['train_ret']=1.0
            label_0['train_ret']=-1.0
            feature_one=pd.concat([label_1,label_0]).iloc[:,2:-2].values.tolist()
            for i in range(len(feature_one)):
                features.append(feature_one[i])
            label_one=pd.concat([label_1,label_0])['train_ret'].values.tolist()
            for i in range(len(label_one)):
                labels.append(label_one[i])
        # train
        classifierArr = adaBoostTrainDS(mat(features), labels, 30)
        predict_data=get_factor(trading_day)
        x=predict_data.iloc[:,2:].values.tolist()
        
        y=adaClassify(x,classifierArr)
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


# In[ ]:

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# 提取交易日
trading_date=DataAPI.TradeCalGet(exchangeCD="XSHG",beginDate="20100101",endDate="20160101",field=["calendarDate","isMonthEnd","isOpen"],pandas="1")
trading_date_open=trading_date[trading_date['isOpen'] == 1]
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
        
        month_closePrice=pd.read_csv('buy_price_one'+trading_day+'.csv')
        next_month_closePrice=pd.read_csv('next_price_one'+next_trading_day+'.csv')
        month_data=pd.DataFrame(columns=['CurrentPrice','NextPrice','Return'])
        month_data['CurrentPrice']=month_closePrice['closePrice'].values
        month_data['NextPrice']=next_month_closePrice['closePrice'].values
        month_data['Return']=(month_data['NextPrice']-month_data['CurrentPrice'])/month_data['CurrentPrice']
        mean_ret=np.mean(month_data['Return'])
        capital=capital*(mean_ret+1)
                
        benchMark=DataAPI.MktIdxdGet(tradeDate=trading_day,ticker="000300",field='ticker,tradeDate,closeIndex')
        bmprice = benchMark['closeIndex'].values
        bp.append(bmprice)
        ret.append(capital)
rb = [bp[index][0]/bp[0][0] for index in range(len(bp))] 
ret_rb=pd.DataFrame(index=monthend_day['calendarDate'].values[:-1],columns=['ret','rb'])
ret_rb['ret']=ret
ret_rb['rb']=rb
ret_rb.plot(figsize=(16,8))


# In[ ]:


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)#call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        #print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        #print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        #print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        #print "total error: ",errorRate
        if errorRate == 0.0: break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])#call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
    return aggClassEst


# In[ ]:

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


# In[ ]:

a=list_files()
zip_files('data_all', a)


# In[ ]:



