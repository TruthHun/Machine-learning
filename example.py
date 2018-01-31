# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from datetime import datetime
import numpy as np
from gm.api import *
import sys
try:
    from sklearn import svm
except:
    print('请安装scikit-learn库和带mkl的numpy')
    sys.exit(-1)

'''
本策略选取了七个特征变量组成了滑动窗口长度为15天的训练集,随后训练了一个二分类(上涨/下跌)的支持向量机模型.
若没有仓位则在每个星期一的时候输入标的股票近15个交易日的特征变量进行预测,并在预测结果为上涨的时候购买标的.
若已经持有仓位则在盈利大于10%的时候止盈,在星期五损失大于2%的时候止损.
特征变量为:1.收盘价/均值2.现量/均量3.最高价/均价4.最低价/均价5.现量6.区间收益率7.区间标准差
训练数据为:SHSE.600009上海机场,时间从2016-04-01到2017-07-30
回测时间为:2017-08-01 09:00:00到2017-09-05 09:00:00
'''


def init(context):
    # 订阅上海机场的分钟bar行情
    context.symbol = 'SHSE.600009'
    subscribe(symbols=context.symbol, frequency='60s')
    start_date = '2016-03-01'  # SVM训练起始时间
    end_date = '2017-06-30'  # SVM训练终止时间
    # 用于记录工作日
    # 获取目标股票的daily历史行情
    recent_data = history(symbol=context.symbol, frequency='1d', start_time=start_date, end_time=end_date, fill_missing='Last',
                          df=True)
    days_value = recent_data['bob'].values
    days_close = recent_data['close'].values
    days = []
    # 获取行情日期列表
    print('准备数据训练SVM')
    for i in range(len(days_value)):
        days.append(str(days_value[i])[0:10])

    x_all = []
    y_all = []
    for index in range(15, (len(days) - 5)):
        # 计算三星期共15个交易日相关数据
        start_day = days[index - 15]
        end_day = days[index]
        data = history(symbol=context.symbol, frequency='1d', start_time=start_day, end_time=end_day, fill_missing='Last',
                       df=True)
        close = data['close'].values
        max_x = data['high'].values
        min_n = data['low'].values
        amount = data['amount'].values
        volume = []
        for i in range(len(close)):
            volume_temp = amount[i] / close[i]
            volume.append(volume_temp)

        close_mean = close[-1] / np.mean(close)  # 收盘价/均值
        volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
        max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
        min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
        vol = volume[-1]  # 现量
        return_now = close[-1] / close[0]  # 区间收益率
        std = np.std(np.array(close), axis=0)  # 区间标准差

        # 将计算出的指标添加到训练集X
        # features用于存放因子
        features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
        x_all.append(features)

    # 准备算法需要用到的数据
    for i in range(len(days_close) - 20):
        if days_close[i + 20] > days_close[i + 15]:
            label = 1
        else:
            label = 0
        y_all.append(label)

    x_train = x_all[: -1]
    y_train = y_all[: -1]
    # 训练SVM
    context.clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                          tol=0.001, cache_size=400, verbose=False, max_iter=-1,
                          decision_function_shape='ovr', random_state=None)
    context.clf.fit(x_train, y_train)
    print('训练完成!')


def on_bar(context, bars):
    bar = bars[0]
    # 获取当前年月日
    today = bar.bob.strftime('%Y-%m-%d')
    last_day = get_previous_trading_date(exchange='SHSE', date=today)
    # 获取数据并计算相应的因子
    # 于星期一的09:31:00进行操作
    # 当前bar的工作日
    weekday = datetime.strptime(today, '%Y-%m-%d').isoweekday()
    # 获取模型相关的数据
    # 获取持仓
    position = context.account().position(symbol=context.symbol, side=PositionSide_Long)
    # 如果bar是新的星期一且没有仓位则开始预测
    if not position and weekday == 1:
        # 获取预测用的历史数据
        data = history_n(symbol=context.symbol, frequency='1d', end_time=last_day, count=15,
                         fill_missing='Last', adjust=ADJUST_PREV, df=True)
        close = data['close'].values
        train_max_x = data['high'].values
        train_min_n = data['low'].values
        train_amount = data['amount'].values
        volume = []
        for i in range(len(close)):
            volume_temp = train_amount[i] / close[i]
            volume.append(volume_temp)

        close_mean = close[-1] / np.mean(close)
        volume_mean = volume[-1] / np.mean(volume)
        max_mean = train_max_x[-1] / np.mean(train_max_x)
        min_mean = train_min_n[-1] / np.mean(train_min_n)
        vol = volume[-1]
        return_now = close[-1] / close[0]
        std = np.std(np.array(close), axis=0)

        # 得到本次输入模型的因子
        features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
        features = np.array(features).reshape(1, -1)
        prediction = context.clf.predict(features)[0]
        # 若预测值为上涨则开仓
        if prediction == 1:
            # 获取昨收盘价
            context.price = close[-1]
            # 把浦发银行的仓位调至95%
            order_target_percent(symbol=context.symbol, percent=0.95, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print(context.symbol, '以市价单开多仓到仓位0.95')
    # 当涨幅大于10%,平掉所有仓位止盈
    elif position and bar.close / context.price >= 1.10:
        order_close_all()
        print(context.symbol, '以市价单全平多仓止盈')
    # 当时间为周五并且跌幅大于2%时,平掉所有仓位止损
    elif position and bar.close / context.price < 1.02 and weekday == 5:
        order_close_all()
        print(context.symbol, '以市价单全平多仓止损')


if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='strategy_id',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='token_id',
        backtest_start_time='2017-08-01 09:00:00',
        backtest_end_time='2017-09-05 09:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)