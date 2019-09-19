import pandas as pad
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm, model_selection #model_selection replaced cross validation
from sklearn.model_selection import cross_validate, train_test_split #just in case, I imported these as well
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = '9U6i-WCqzgcfFq86cQ93'#my quandl key

df = quandl.get('WIKI/GOOGL')#can be changed to any other stock

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]    #creating our own chart
df['HL_PCT']= (df['Adj. High']-df['Adj. Close']) / df['Adj. Close'] *100.0  #calculating high-low percent (percent volatility)
df['PCT_change']= (df['Adj. Close']-df['Adj. Open']) / df['Adj. Open'] *100.0 #daily percent move/change

#            price       x              x           x
df = df[['Adj. Close', 'HL_PCT', 'PCT_change','Adj. Volume']] #this is the columns we want to print

forecast_col = 'Adj. Close' #new column for storing prediction
df.fillna(-99999, inplace=True) #fill not available spaces w an outlier, because can not work with nonexistent data

forecast_out = int(math.ceil(0.01*len(df))) #going to try to predict 1% of the data frame out by 1 day
#print(forecast_out) #35 data points

df['label']= df[forecast_col].shift(-forecast_out)  #label is the forecast col shifting up.. each row will be the adjusted close price 1 day into the future



X = np.array(df.drop(['label', 'Adj. Close'],1)) #X is feature (input)
X = preprocessing.scale(X) #data is very spread out, so might be skewed. scaling values normalizes it
X_lately = X[-forecast_out:] # ":" means to the point of... this reads X to the point of negative forecast_out
X = X[:-forecast_out] #X_lately is what we predict against, the last 35 days



df.dropna(inplace=True)
y = np.array(df['label'])   #y is label (output)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1) #clf means classifier. better algorithm. in () can add n_jobs to have parallel threads
#clf = svm.SVR()    #worse algorithm
clf.fit(X_train, y_train) #fit is basically training
with open('linearregression.pickle', 'wb') as f:
    pickle.dump(clf, f) #dumps classifier into f

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)    #score is basically testing
#print(accuracy) #how close is our test (X_test, y_test) to the actual data
forecast_set = clf.predict(X_lately)    #*IMPORTANT*can pass 1 value or an array of values to predict per value in that array. We use X_lately bc it's last 35 days
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()














