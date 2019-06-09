#%% packages

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
#import matplotlib.dates as md
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import xgboost
import datetime as dt
import statsmodels.api as sm
import itertools
plt.style.use('seaborn-white')
from statsmodels.tsa.arima_model import ARIMA
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
print(tf.__version__)
print(keras.__version__)

np.random.seed(1337)
PYTHONHASHSEED = 0
tf.random.set_random_seed(1337)
tf.set_random_seed(1337)

featSelect = 0

#%% load data
finalDataDf = pd.read_csv('data/MyData12.csv')
finalData = finalDataDf.values
X = finalData[:,:-1]
y = finalData[:,-1]


#%% feature normalization  and preprocess
testRate = 0.2
numTrain = int(len(X) * (1-testRate))
numTest = 1

m = X[:numTrain].mean(axis=0)
s = X[:numTrain].std(axis=0)

X = ( X - m ) / s
X[:,0:2] = 0
realX = X
#X_train = ( X_train - m ) / s
#X_test = ( X_test - m ) / s

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testRate, shuffle = False)

numTest = len(y_test)
#numTrain = 1 - numTest

vec_date = np.vectorize(dt.date)

t = vec_date(finalData[:,0], finalData[:,1], 1)
#t_test = vec_date(X_test[:, 0], X_test[:, 1], 1)
t_test = vec_date(finalData[numTrain:, 0], finalData[numTrain:, 1], 1)
allFig = plt.figure()
allFigax = allFig.add_subplot(111)
allFigax.plot_date(t, y, '-', label='actual')
allFigax.legend(loc='upper right')
allFigax.set_xlabel("time")
allFigax.set_ylabel('volume(m3)')



#%% feature selection

lassoIC = LassoLarsIC()
lassoIC.fit(X_train, y_train)
coef = lassoIC.coef_
X = X[:,coef!=0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

coef = np.append(coef,0)
print(finalDataDf.columns[coef!=0])

#di = {'feature':finalDataDf.columns[coef!=0], 'coef':coef[coef!=0]}
#selectedFeature = pd.DataFrame(data=di)

selectedFeature = pd.DataFrame(data=coef[coef!=0], index=finalDataDf.columns[coef!=0], columns=['coef'])
selectedFeature.sort_values(by=['coef'],ascending=0).to_csv("featureCoef.csv")




#%% time series

realY = y

y = pd.DataFrame(y_train)
y.index = pd.DatetimeIndex(t[:numTrain])
decomposition = sm.tsa.seasonal_decompose(y[0], model='additive')

fig = decomposition.plot()
plt.xlabel("time")


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

Param = []
Param_seasonal = []
AIC = []

#%% SARIMAX
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            Param.append(param)
            Param_seasonal.append(param_seasonal)
            AIC.append(results.aic)
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        
minAIC = AIC.index(min(AIC))

mod = sm.tsa.statespace.SARIMAX(y,
                                order=Param[minAIC],
                                seasonal_order=Param_seasonal[minAIC],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred_uc = results.get_forecast(steps=numTest)
pred_ci = pred_uc.conf_int()
#ax = yttrain.plot(label='observed', figsize=(14, 7))
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()
plt.show()

predictions = pred_uc.predicted_mean.values

#mse = ((predictions - y_test) ** 2).mean()
np.mean(abs(predictions - y_test)/y_test)
#
#print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))




#%% ARIMA   
Param = []

AIC = []     
for param in pdq:
    
    try:
        
        mod = ARIMA(y, order=param)
        results = mod.fit()
        Param.append(param)
        
        AIC.append(results.aic)
        print('ARIMA{} - AIC:{}'.format(param, results.aic))
    except:
        continue

minAIC = AIC.index(min(AIC))

mod = ARIMA(y,order=Param[minAIC])
results = mod.fit()
print(results.summary().tables[1])    

predictions = results.forecast(steps=numTest)[0]
#predictions = results.predict(start=dt.datetime(X_test[0,0], X_test[0,1], 1),
#                end=dt.datetime(X_test[-1,0], X_test[-1,1], 1))
np.mean(abs(predictions - y_test)/y_test)

#%% test plot

fig = plt.figure()
ax = fig.add_subplot(211)
plt.plot_date(t_test, y_test,'-', label='actual')
plt.legend(loc='upper right')

plt.ylabel('water supply(t)')
plt.title('Water Supply Prediction')

predAll = []

ax2 = fig.add_subplot(212)
plt.plot_date(t_test, y_test-y_test,'-', label='actual')
plt.grid(axis="y")
plt.ylabel('error(t)')
plt.xlabel('time')

def draw_prediction(predictions, y_test, mName):
    
    res = predictions - y_test
    ax.plot_date(t_test, predictions,'-', label=mName)
    ax.legend(loc='best')
    ax2.plot_date(t_test, res, '-', label=mName)
    ax2.legend(loc='best')
    
    allFigax.plot_date(t_test, predictions,'-', label=mName)
    allFigax.legend(loc='best')
    # np.mean(abs(y_train_pred - y_train))/np.mean(y_train)
    print(np.mean(abs(res/y_test)))
    # print(np.mean(abs(res))/np.mean(y_test))




#%% y normalization

mY = y_train.mean()
sY = y_train.std()
y_train = ( y_train - mY ) / sY
# y_test = ( y_test - mY ) / sY



#%% lasso regression
"""
lassocv.alplha_ is different from which in R
"""

mName = 'lasso'
lassocv = LassoLarsIC()
lassocv.fit(X_train, y_train)
y_train_pred = lassocv.predict(X_train)
predictions = lassocv.predict(X_test)
predictions = predictions * sY + mY
#predAll = np.append(predAll,predictions).reshape([-1,1])
coef = lassocv.coef_
lassocv.alpha_

draw_prediction(predictions, y_test, mName)

## lassocv = LassoCV(random_state=0, eps=1e-9, cv=10, n_alphas=100)
#lassocv = LassoLarsCV()
#lassocv.fit(X_train, y_train)
#y_train_pred = lassocv.predict(X_train)
#predictions = lassocv.predict(X_test)
#np.mean(abs(y_train_pred - y_train))/np.mean(y_train)
#np.mean(abs(predictions - y_test))/np.mean(y_test)
#lassocv.coef_
#lassocv.alpha_
#plt.plot_date(t, predictions,'-', label='lassocv')
#plt.legend(loc='upper right')

#%% Ridge regression
mName = 'Ridge'
lassocv = RidgeCV()
lassocv.fit(X_train, y_train)
coef = lassocv.coef_

y_train_pred = lassocv.predict(X_train)
predictions = lassocv.predict(X_test)
predictions = predictions * sY + mY
#predAll = np.append(predAll,predictions.reshape([-1,1]), axis=1)

coef = lassocv.coef_
lassocv.alpha_

draw_prediction(predictions, y_test, mName)

# ax.plot_date(t_test, predictions,'-', label=mName)
# ax.legend(loc='best')
# res = predictions - y_test
# ax2.plot_date(t_test, res, '-', label=mName)
# ax2.legend(loc='best')
# np.mean(abs(y_train_pred - y_train))/np.mean(y_train)
# np.mean(abs(predictions - y_test))/np.mean(y_test)

#%% Random Forest
X = realX
y = realY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
y_train = ( y_train - mY ) / sY
mName = 'Random Forest'
regr = RandomForestRegressor(random_state=1337)
regr.fit(X_train, y_train)



temp = regr.feature_importances_
temp.sort()
coef = regr.feature_importances_
#coef = np.append(coef,0)

if featSelect == 1:
    
    X = X[:,coef>=temp[-3]]
    y = realY
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
    coef = np.append(coef,0)
    y_train = ( y_train - mY ) / sY
    
    
    selectedFeature = pd.DataFrame(data=coef[coef>=temp[-3]], index=finalDataDf.columns[coef>=temp[-3]], columns=['coef'])
    selectedFeature.sort_values(by=['coef'],ascending=0).to_csv("featureCoef.csv")
    regr = RandomForestRegressor(random_state=1337)
    regr.fit(X_train, y_train)


y_train_pred = regr.predict(X_train)
predictions = regr.predict(X_test)
predictions = predictions * sY + mY
#predAll = np.append(predAll,predictions.reshape([-1,1]), axis=1)

draw_prediction(predictions, y_test, mName)

# ax.plot_date(t_test, predictions, '-', label=mName)
# ax.legend(loc='best')

# res = predictions - y_test
# ax2.plot_date(t_test, res, '-', label=mName)
# ax2.legend(loc='best')
# np.mean(abs(y_train_pred - y_train))/np.mean(y_train)
# np.mean(abs(predictions - y_test))/np.mean(y_test)

#%% XGBoost
X = realX
y = realY
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
y_train = ( y_train - mY ) / sY
mName = 'XGBoost'
xgb = xgboost.XGBRegressor(seed=1337)
xgb.fit(X_train, y_train, verbose=True)

temp = xgb.feature_importances_
temp.sort()
coef = xgb.feature_importances_

if featSelect == 1:

    X = X[:,coef>=temp[-3]]
    y = realY
    coef = np.append(coef,0)
    selectedFeature = pd.DataFrame(data=coef[coef>=temp[-3]], index=finalDataDf.columns[coef>=temp[-3]], columns=['coef'])
    selectedFeature.sort_values(by=['coef'],ascending=0).to_csv("featureCoef.csv")

    
    # 4 feature selected rather than 3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
    
    y_train = ( y_train - mY ) / sY
    
    xgb = xgboost.XGBRegressor()
    xgb.fit(X_train, y_train)


y_train_pred = xgb.predict(X_train)
predictions = xgb.predict(X_test)
predictions = predictions * sY + mY
#predAll = np.append(predAll,predictions.reshape([-1,1]), axis=1)

draw_prediction(predictions, y_test, mName)

#%% y normalization

#mY = y_train.mean()
#sY = y_train.std()
#y_train = ( y_train - mY ) / sY
#y_test = ( y_test - mY ) / sY

#%% NN
mName = 'NN'
val_loss = []
model_rec = []
history_rec = []

for i in range(10):
    model = Sequential()
    
    model.add(Dense(units=3, activation='relu', input_dim=X.shape[1]))
    #model.add(Dense(units=3, activation='relu', input_dim=X.shape[1]))
    #model.add(Dense(units=3, activation='relu', input_dim=X.shape[1]))
    #model.add(Dense(units=3, activation='relu', input_dim=X.shape[1]))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1))
    model.summary()
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
#    model.save_weights('model.h5')
    
    
    model_rec.append(model)
    
    patience = 200
    es=keras.callbacks.EarlyStopping(monitor='val_loss',
                                     mode='min',
                                     patience=patience)
    history = model.fit(X_train, y_train, epochs=20000, validation_split=0.2, callbacks=[es])
    
    history_rec.append(history)
    val_loss.append(history.history['val_loss'][-1])

   
model = model_rec[val_loss.index(min(val_loss))]
history = history_rec[val_loss.index(min(val_loss))]

#epochs = epoch_rec[val_loss.index(min(val_loss))] - patience
epochs = len(history.epoch) - patience
#model.load_weights('model.h5')
model.fit(X_train, y_train, epochs=epochs)


predictions = model.predict(X_test) * sY + mY
predictions = predictions.reshape(-1)
#predAll = np.append(predAll,predictions.reshape([-1,1]), axis=1)
res = predictions - y_test

draw_prediction(predictions, y_test, mName)
model.save_weights(mName + '.h5')

fig = plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()



#%% LSTM
mName = 'LSTM'
val_loss = []
model_rec = []
history_rec = []

X_train = X_train.reshape(-1,1,X.shape[1])
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1,1,X.shape[1])
    

for i in range(10):
        
    model = Sequential()
    model.add(keras.layers.LSTM(units=3, input_shape=(1,X.shape[1]), activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=3, activation='relu'))
    model.add(Dense(units=1))
    
    model.summary()
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    
    model.save_weights('model' + str(i) + '.h5')
    model_rec.append(model)

    #model.fit(X_train, y_train,epochs=100)
    
    
    patience = 200
    es=keras.callbacks.EarlyStopping(monitor='val_loss',
                                     mode='min',
                                     patience=patience)
    history = model.fit(X_train, y_train, epochs=20000, validation_split=0.2, callbacks=[es])

    history_rec.append(history)
    val_loss.append(history.history['val_loss'][-1])

model.load_weights('model' + str(val_loss.index(min(val_loss))) + '.h5')
#model = model_rec[val_loss.index(min(val_loss))]
history = history_rec[val_loss.index(min(val_loss))]

epochs = len(history.epoch) - patience
#model.load_weights('model.h5')

model.fit(X_train, y_train, epochs=epochs)
#X_test_comb = X_test
X_test_comb = np.concatenate((X_train, X_test))

predictions = model.predict(X_test_comb) * sY + mY
predictions = predictions.reshape(-1)
predictions = predictions[-len(y_test):]
#predAll = np.append(predAll,predictions.reshape([-1,1]), axis=1)
res = predictions - y_test

draw_prediction(predictions, y_test, mName)
model.save_weights(mName + '.h5')

fig = plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
