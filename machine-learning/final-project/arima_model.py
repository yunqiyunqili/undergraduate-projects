# 导入库
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import sqrt
from sklearn.metrics import mean_squared_error

# 样式设置
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

# 读取数据
data = pd.read_csv('data1.csv', engine='python')
# A bit of pre-processing to make it nicer
data['Month']=pd.to_datetime(data['Month'], format='%Y-%m-%d')
data.set_index(['Month'], inplace=True)

# 绘制图
data.plot()
plt.ylabel('Monthly Temp')
plt.xlabel('Date')
plt.show()

# 定义d和q参数为0到1之间的任意值
q = d = range(0, 2)
# 定义p参数为0到3之间的任意值
p = range(0, 4)

# 生成p d q三元组的所有不同组合
pdq = list(itertools.product(p, d, q))

# 生成所有不同的季节性p, d和q三元组的组合
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

train_data = data['2001-01-01':'2011-12-01']
test_data = data['2012-01-01':'2012-12-01']

warnings.filterwarnings("ignore") # specify to ignore warning messages

AIC = []
SARIMAX_model = []
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
            AIC.append(results.aic)
            SARIMAX_model.append([param, param_seasonal])
        except:
            continue

print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))

# Let's fit this model
mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

results.plot_diagnostics(figsize=(20, 14))
plt.show()

pred0 = results.get_prediction(start='2010-01-01', dynamic=False)
pred0_ci = pred0.conf_int()

pred1 = results.get_prediction(start='2010-01-01', dynamic=True)
pred1_ci = pred1.conf_int()

pred2 = results.get_forecast('2013-12-01')
pred2_ci = pred2.conf_int()
print(pred2.predicted_mean)

ax = data.plot(figsize=(20, 16))
pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
plt.ylabel('Monthly Temp')
plt.xlabel('Date')
plt.legend()
plt.show()

prediction = pred2.predicted_mean['2012-01-01':'2012-12-01'].values
# 扁平嵌套列表
truth = list(itertools.chain.from_iterable(test_data.values))
# MAPE
MAPE = np.mean(np.abs((truth - prediction) / truth)) * 100
mse = sqrt(mean_squared_error(truth, prediction))

print('The Mean Absolute Percentage Error for the forecast of year 2012 is {:.2f}%'.format(MAPE))
print('The Mean Square Error for the forecast of year 2012 is {:.2f}'.format(mse))