#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:44:45 2017

@author: bhaskar
"""

import numpy as np
from math import sqrt
#import statsmodels.api as sm
from statsmodels.tsa.api import VAR
#from pandas import Series
from matplotlib import pyplot
#from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pandas import read_csv
from pandas import DataFrame

dframe=DataFrame()
for i in range(15):
    dframe=dframe.append(read_csv('Data/'+str(2000+i)+'.csv',skiprows=[0,1]),ignore_index=True)

train=np.zeros(shape=(365*12,24),dtype='float')
test=np.zeros(shape=(365*3,24),dtype='float')
for i in range(365*12):
    train[i,:]=dframe['GHI'][i*24:(i+1)*24]
for i in range(365*3):
    test[i,:]=dframe['GHI'][i*24+12*365*24:(i+1)*24+12*365*24]    

model = VAR(train)
#results=model.fit(trend='nc',ic=None)
results=model.fit(5, trend='nc')

params=results.params
lag=results.k_ar


#np.dot(train[1,:],params[0:24,:])+np.dot(train[0,:],params[24:48,:])  
    
n_pred=len(test)-lag
predictions=np.zeros(shape=(n_pred,24),dtype=float)


for i in range(n_pred):
    for j in range(lag):
        predictions[i,:]+=np.dot(test[i+j,:],params[24*(lag-j-1):24*(lag-j),:])



average=np.zeros(shape=(365,24),dtype=float)
for i in range(365):
    for k in range(12):
            average[i,:]+=train[365*k+i,:]
average/=12

predictions2=np.zeros(shape=(n_pred,24),dtype=float)

for i in range(n_pred):
    predictions2[i,:]=average[(i+lag)%365,:]
    
mae_var=0    
mae_avg=0
mse_var=0
mse_avg=0
for i in range(n_pred):
    mae_var += mean_absolute_error(predictions[i,:],test[i+lag,:])
    mse_var += mean_squared_error(predictions[i,:],test[i+lag,:])
    mae_avg += mean_absolute_error(predictions2[i,:],test[i+lag,:])
    mse_avg += mean_squared_error(predictions2[i,:],test[i+lag,:])
mae_var/=n_pred
mse_var=sqrt(mse_var/n_pred)
mae_avg/=n_pred
mse_avg=sqrt(mse_avg/n_pred)

       
print('VAR Test MAE: %.3f MSE:%.3f' %(mae_var, mse_var))
print('AVG Test MAE: %.3f MSE:%.3f' %(mae_avg, mse_avg))
# plot results

#for hour in range(5,18):
#    t=365
#    t1=range(0,t)
#    t2=range(0+lag,t+lag)
#    pyplot.plot(test[t2,hour],color='blue',label='actual')
#    pyplot.plot(predictions[t1,hour], color='red',label='VAR')
#    pyplot.plot(predictions2[t1,hour], color='yellow',label='avg')
#    pyplot.legend(loc='lower right')
#    pyplot.title('Yearly GHI at '+str(hour)+': 30')
#    pyplot.axis([0,t,0,1000])
#    pyplot.savefig('plots/'+str(hour)+'_30')
#    pyplot.show()

start_day=300
end_day=310
for i in range(start_day,end_day):
    pyplot.plot(test[i+lag,:],color='blue',label='Actual')
    pyplot.plot(predictions[i,:], color='red',label='VAR')
    pyplot.plot(predictions2[i,:], color='yellow',label='Avg')
    pyplot.legend(loc='upper right')
    pyplot.axis([0,23,0,1000])
    pyplot.title('day_'+str(i))
    pyplot.savefig('plots/day_'+str(i))
    pyplot.show()

