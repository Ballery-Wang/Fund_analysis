# Fund_analysis
主要用于基金净值的各种分析
input:
    funddata: 基金净值数据，Series类型的时间序列
    fundname: 基金名称，字符串格式
    benchmark: 对标基准，可以传入三种类型
        1.不传入参数(默认),代表没有基准
        2.传入Wind指数代码，指定对标基准，字符串格式，Wind指数代码字典如下：
        {'000001.SH':'上证综指','399001.SZ':'深证成指','399101.SZ':'中小板综指','399006.SZ':'创业板指','000016.SH':'上证50','000300.SH':'沪深          300','000905.SH':'中证500','000852.SH':'中证1000',\
         '881001.WI':'Wind全A','H01001.CSI':'中证全债(净价)','NH0100.NHF':'南华商品指数','NH0008.NHF':'南华黄金指数','NH0041.NHF':'南华原油指            数','USDCNY.IB':'美元兑人民币'}
        3.传入用户自己的基准净值序列，Series类型的时间序列，要求index和基金净值一致
    risk_free_r: 无风险收益，默认为0
output:
    计算结果会输出到Excel中，保存路径为当前工作路径下
例子:
    model = Fund_analysis(funddata=fund1,fundname='九坤量化2号',benchmark='881001.WI')
    model.cal_score()
    model.save_output()
