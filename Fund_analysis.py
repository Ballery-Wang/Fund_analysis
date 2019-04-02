# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:18:50 2018

@author: Ballery Wang
基金净值分析
"""

import os
import pandas as pd
import numpy as np
import math
from sklearn import linear_model
import pylab
import itertools
pylab.mpl.rcParams['font.sans-serif'] = ['SimHei'] # 载入中文字体，画图时使用
import datetime

class Fund_analysis():
    '''
    该类主要用于基金净值的各种分析
    input:
        funddata: 基金净值数据，Series类型的时间序列
        fundname: 基金名称，字符串格式
        benchmark: 对标基准，可以传入三种类型
            1.不传入参数(默认),代表没有基准
            2.传入Wind指数代码，指定对标基准，字符串格式，Wind指数代码字典如下：
            {'000001.SH':'上证综指','399001.SZ':'深证成指','399101.SZ':'中小板综指','399006.SZ':'创业板指','000016.SH':'上证50','000300.SH':'沪深300','000905.SH':'中证500','000852.SH':'中证1000',\
            '881001.WI':'Wind全A','H01001.CSI':'中证全债(净价)','NH0100.NHF':'南华商品指数','NH0008.NHF':'南华黄金指数','NH0041.NHF':'南华原油指数','USDCNY.IB':'美元兑人民币'}
            3.传入用户自己的基准净值序列，Series类型的时间序列，要求index和基金净值一致
        risk_free_r: 无风险收益，默认为0
    output:
        计算结果会输出到Excel中，保存路径为当前工作路径下
    例子:
        model = Fund_analysis(funddata=fund1,fundname='九坤量化2号',benchmark='881001.WI')
        model.cal_score()
        model.save_output()
    '''
    def __init__(self,funddata,fundname='X基金',benchmark=None,risk_free_r=0):
        # 构造基金净值
        if isinstance(funddata,pd.Series) :
            self.funddata = funddata.astype(np.float64).sort_index()
            if isinstance(self.funddata.index[0],datetime.date) :
                pass
            elif isinstance(self.funddata.index[0],datetime.datetime) :
                self.funddata.index = self.funddata.index.map(lambda x : x.date())
            else :
                print('传入的基金净值数据要求是时间序列,index需为日期格式')
        else :
            print('传入的基金净值数据要求是Series类型')
        self.fundname = fundname
        # 构造基准
        index_dict = {'000001.SH':'上证综指','399001.SZ':'深证成指','399101.SZ':'中小板综指','399006.SZ':'创业板指','000016.SH':'上证50','000300.SH':'沪深300','000905.SH':'中证500','000852.SH':'中证1000','881001.WI':'Wind全A','H01001.CSI':'中证全债(净价)','NH0100.NHF':'南华商品指数','NH0008.NHF':'南华黄金指数','NH0041.NHF':'南华原油指数','USDCNY.IB':'美元兑人民币'}
        if benchmark is None :
            self.benchmark = pd.Series(1,index=self.funddata.index)
            self.ben_name = '无基准'
        elif isinstance(benchmark,str) :
            self.benchmark = benchmark
            self.ben_name = index_dict[benchmark]
        elif isinstance(benchmark,pd.Series) :
            if benchmark.index is funddata.index:
                self.benchmark = benchmark.astype(np.float64).sort_index()
                self.ben_name = '自定义基准'
            else :
                print('传入的业绩基准的索引需与基金净值相同')
        else :
            print('传入的 benchmark 格式错误 ')
        self.risk_free_r = risk_free_r
       
    # 获得基金运行时长（单位：年）
    def get_days(self):
        self.start_day = self.funddata.index[0]
        self.end_day = self.funddata.index[-1]
        self.days = self.end_day - self.start_day
        self.years = self.days.days/365.24
        # 统计每年公布净值的期数
        self.freq = self.funddata.shape[0] / self.years
    
    # 获取对标基准数据
    def get_benchmark(self):
        import WindPy as wind
        wind.w.start()
        import datetime
        self.benchmark = wind.w.wsd(self.benchmark, "close", self.start_day.strftime('%Y-%m-%d'), self.end_day.strftime('%Y-%m-%d'), "Days=Alldays",usedf = True)[1]
        self.benchmark = self.benchmark.reindex(self.funddata.index)
        self.benchmark = self.benchmark.CLOSE
        
    # 净值归一处理，基金净值与基准对比基准和超额收益数据
    def Normalization(self):
        self.funddata = self.funddata/self.funddata[0]
        self.benchmark = self.benchmark/self.benchmark[0]
        self.excess = self.funddata - self.benchmark
        self.show_data = pd.DataFrame({self.fundname:self.funddata,self.ben_name:self.benchmark,'超额收益':self.excess})

    # 计算年化收益率
    def get_Annual_earnings(self):
        self.profit = self.funddata[-1]/self.funddata[0] - 1 # 基金总收益
        self.ann_return = math.pow(self.profit + 1,1/self.years)-1
    
    # 计算基准年化收益率
    def get_B_Annual_earnings(self):
        self.b_profit = self.benchmark.values[-1]/self.benchmark.values[0] - 1 # 基准总收益
        self.ben_ann_return = math.pow(self.b_profit + 1,1/self.years)-1
        
    # 计算最大回撤和最大回撤发生的日期
    def maxDrawdown(self):
        temp_01 = np.maximum.accumulate(self.funddata)
        drawdown = (temp_01 - self.funddata)/temp_01
        self.max_drawdown = drawdown.max()
        self.max_draw_date = drawdown.idxmax()

    # 计算相对收益的最大回撤和最大回撤发生的日期
    def maxDrawdown_rel(self):     
        temp_01 = np.maximum.accumulate(self.excess+1)
        drawdown = (temp_01 - (self.excess+1) )/temp_01
        self.max_drawdown_rel = drawdown.max()
        self.max_draw_date_rel = drawdown.idxmax()
        
    # 计算年化标准差(周度净值频率)
    def annual_standard(self):
        self.returns = self.funddata.pct_change().dropna()
        self.ann_std = self.returns.std() * np.sqrt(self.freq)
    
    # 计算夏普比率
    def sharpe_ratio(self):
        self.sharpe = (self.ann_return - self.risk_free_r) / self.ann_std
    
    # 计算卡玛比率
    def calmar_Ratio(self):
        self.calmar = self.ann_return/self.max_drawdown
        #计算超额收益的卡玛比率  
        self.calmar_rel = (self.ann_return - self.ben_ann_return)/self.max_drawdown_rel
    
    # 计算alpha和beta
    def alpha_beta(self):
        self.b_returns = self.benchmark.pct_change().dropna()
        reg = linear_model.LinearRegression()
        reg.fit(self.b_returns.values.reshape(-1,1),self.returns.values)
        self.alpha = reg.intercept_
        self.beta = reg.coef_[0]
    
    # 计算特雷诺比率
    def treynor_ratio(self):
        self.treynor = (self.ann_return - self.risk_free_r) / self.beta

    # 计算下行标准差
    def downside_risk(self):
        returns = self.returns.copy()
        returns[returns > 0] = 0
        self.down_risk = returns.std() * np.sqrt(self.freq)
    
    # 计算索提诺比率
    def sortino_Ratio(self):
        self.sortino = (self.ann_return - self.risk_free_r) / self.down_risk
        
    # 计算信息比率
    def information_Ratio(self):
        self.excess_returns = self.returns - self.b_returns
        self.tracking_error = self.excess_returns.std() * np.sqrt(self.freq)
        self.information = (self.excess_returns.mean())*( self.freq )/self.tracking_error
        
    '''
    统计盈利亏损期占比和最大连涨连跌数
    tot_count:运行总期数
    average_revenue:平均收益
    profit_count:盈利期数
    profit_ratio:盈利期占比
    loss_count:亏损期数
    loss_ratio:亏损期占比
    max_con_rise:最大连涨期数
    max_con_decline:最大连跌期数
    '''
    def profit_loss(self):
        self.tot_count = len(self.returns)
        self.average_revenue = self.profit/self.tot_count
        self.profit_count = len(self.returns[self.returns>0])
        self.loss_count = len(self.returns[self.returns<0])
        self.profit_ratio = self.profit_count/self.tot_count
        self.loss_ratio = self.loss_count/self.tot_count
        # 求最大连续上涨和下跌次数
        uad = np.sign(self.returns)
        temp = [[k,len(list(v))] for k,v in itertools.groupby(uad.values)]
        temp2 = pd.DataFrame(temp)
        self.max_con_rise = temp2[temp2[0]==1][1].max()
        self.max_con_decline = temp2[temp2[0]==-1][1].max()
    
    # 统计收益区间
    def income_interval(self):
        # 近一月收益
        date_node = self.funddata.index[-1] + datetime.timedelta(-30)
        self.l_month = self.funddata[-1]/self.funddata.loc[date_node:][0] - 1
        # 近三月收益
        date_node = self.funddata.index[-1] + datetime.timedelta(-91)
        self.l_3month = self.funddata[-1]/self.funddata.loc[date_node:][0] - 1
        # 近半年收益
        date_node = self.funddata.index[-1] + datetime.timedelta(-182)
        self.h_year = self.funddata[-1]/self.funddata.loc[date_node:][0] - 1
        # 年初至今收益
        date_node = datetime.date(self.funddata.index[-1].year,1,1)
        self.ytd = self.funddata[-1]/self.funddata.loc[date_node:][0] - 1
        # 近一年收益
        date_node = self.funddata.index[-1] + datetime.timedelta(-365)
        self.l_year = self.funddata[-1]/self.funddata.loc[date_node:][0] - 1
    
    # 统计滚动收益(年度),以及收益概率分布
    def rolling_revenue(self):
        date_node = self.funddata.index[0] + datetime.timedelta(365)
        temp1 = self.funddata.loc[date_node:]
        self.y_rolling = pd.Series(index = temp1.index)
        for i in range(len(self.y_rolling)) :
            self.y_rolling.iloc[i] = temp1.iloc[i]/self.funddata.iloc[i] - 1
        self.y_rolling.name = '一年滚动收益'
        # 将滚动年化收益分位十个分位
        quantile = round(self.y_rolling.quantile(np.arange(0.1,1,0.1)),2)
        self.r_prob = pd.DataFrame({'年度滚动收益率' : quantile.map(lambda x: format(x, '.0%')).values,'收益之上发生概率':np.NaN})
        for i in range(self.r_prob.shape[0]) :
            temp2 = quantile.iloc[i]
            temp3 = self.y_rolling[self.y_rolling >= temp2].count()/self.y_rolling.count()
            self.r_prob.iloc[i,1] = format(temp3,'.2%')
    
    # 计算各项指标生成统一输出格式
    def cal_score(self):
        self.get_days()
        if isinstance(self.benchmark,str) :
            self.get_benchmark()
        self.Normalization()
        self.get_Annual_earnings()
        self.get_B_Annual_earnings()
        self.maxDrawdown()
        self.maxDrawdown_rel()
        self.annual_standard()
        self.sharpe_ratio()
        self.calmar_Ratio()
        self.alpha_beta()
        self.treynor_ratio()
        self.downside_risk()
        self.sortino_Ratio()
        self.information_Ratio()
        self.profit_loss()
        self.income_interval()
        self.rolling_revenue()
        self.result = pd.Series(index=['基金运行时长（年）','年化收益率',\
                        '最大回撤','最大回撤发生日期','超额收益最大回撤','超额收益最大回撤发生日期','年化标准差','年化下行标准差','夏普比率',\
                        '卡玛比率','超额收益卡玛比率','特雷诺比率','索提诺比率','信息比率',\
                        '运行总期数','盈利期数','盈利期占比','亏损期数','亏损期占比',\
                        '最大连涨期数','最大连跌期数','平均收益','Alpha','Beta',\
                        '基金总收益','近一月收益','近三月收益','近半年收益','年初至今收益','近一年收益'])
        self.result['基金运行时长（年）'] = round(self.years,1)
        self.result['年化收益率'] = format(self.ann_return, '.2%')
        self.result['最大回撤'] = format(self.max_drawdown, '.2%')
        self.result['最大回撤发生日期'] = self.max_draw_date.strftime('%Y-%m-%d')
        self.result['超额收益最大回撤'] = format(self.max_drawdown_rel, '.2%')
        self.result['超额收益最大回撤发生日期'] = self.max_draw_date_rel.strftime('%Y-%m-%d')
        self.result['年化标准差'] = format(self.ann_std,'.2%')
        self.result['年化下行标准差'] = format(self.down_risk,'.2%')
        self.result['夏普比率'] = round(self.sharpe,2)
        self.result['卡玛比率'] = round(self.calmar,2)
        self.result['超额收益卡玛比率'] = round(self.calmar_rel,2)        
        self.result['特雷诺比率'] = round(self.treynor,2)
        self.result['索提诺比率'] = round(self.sortino,2)
        self.result['信息比率'] = round(self.information,2)
        self.result['运行总期数'] = self.tot_count
        self.result['盈利期数'] = self.profit_count
        self.result['盈利期占比'] = format(self.profit_ratio, '.2%')
        self.result['亏损期数'] = self.loss_count
        self.result['亏损期占比'] = format(self.loss_ratio, '.2%')
        self.result['最大连涨期数'] = self.max_con_rise
        self.result['最大连跌期数'] = self.max_con_decline
        self.result['平均收益'] = format(self.average_revenue, '.2%')
        self.result['Alpha'] = round(self.alpha,4)
        self.result['Beta'] = round(self.beta,4)
        self.result['基金总收益'] = format(self.profit, '.2%')
        self.result['近一月收益'] = format(self.l_month, '.2%')
        self.result['近三月收益'] = format(self.l_3month, '.2%')
        self.result['近半年收益'] = format(self.h_year, '.2%')
        self.result['年初至今收益'] = format(self.ytd, '.2%')
        self.result['近一年收益'] = format(self.l_year, '.2%')
        self.result = pd.DataFrame({self.fundname:self.result})
        print(self.fundname + '指标计算完毕')

    # 部分指标计算，针对只用到少数功能时
    def local_cal(self):
        self.get_days()
        self.annual_standard()
        self.get_Annual_earnings()
        self.sharpe_ratio()
        self.maxDrawdown()
        self.calmar_Ratio()
    
    # 输出结果到当前工作路径下
    def save_output(self):
        save_path = os.path.join(os.getcwd(),'%s分析结果.xlsx'%(self.fundname))
        writer = pd.ExcelWriter(save_path,datetime_format='YYYY-MM-DD')
        self.show_data.to_excel(writer,sheet_name = self.fundname)
        self.result.to_excel(writer,sheet_name = self.fundname,startrow=5,startcol=5)
        self.y_rolling.map(lambda x: format(x,'.2%')).to_excel(writer,sheet_name = '年度滚动收益率')
        self.r_prob.to_excel(writer,sheet_name = '年度滚动收益率',startrow=5,startcol=3)
        writer.save()
        print('分析结果保存至' + os.getcwd() + '路径下')
        
        