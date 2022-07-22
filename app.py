from pickle import FALSE
from flask import Flask,render_template,redirect,request
from flask.helpers import flash
from tempfile import TemporaryDirectory
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.callbacks import EarlyStopping
from numpy import array
from numpy.random import seed
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from prophet.plot import plot_plotly, plot_components_plotly
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import statsmodels.api as sm
import warnings
from werkzeug import debug
warnings.filterwarnings('ignore')
import yfinance as yf
import datetime,math
from datetime import timedelta, date
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
import plotly.express as px

app = Flask(__name__)

    

hist_df = pd.read_csv('USD_INR DATA.csv')
hist_df["Date"] = pd.to_datetime(hist_df["Date"])
#hist_df = hist_df.set_index(["Date"], drop=True)  
hist_df = hist_df.drop(['Open', 'Low', 'High'], axis = 1)
hist_df = hist_df.dropna()
start_date = date(2022, 3, 15)
end_date = datetime.date.today()

gap = end_date-start_date
days=str(gap.days)+'d'
data = yf.download(tickers = 'USDINR=X', period = days, interval = '1d')
data = data.reset_index()
data["Date"] = pd.to_datetime(data["Date"])
res = data[~(data["Date"] < '2022-03-16')]
#res = res.set_index(["Date"], drop=True)
print(res)
edit_data = res.drop(['Open', 'Low', 'High', 'Adj Close', 'Volume'], axis = 1)
usdinr_df = pd.concat([hist_df, edit_data],ignore_index=True)
inr_df = usdinr_df.set_index(["Date"], drop=True)

#print(usdinr_df)
print(inr_df)

length=len(inr_df)
data_day1=inr_df[length-1:]

price_day1=round(float(data_day1['Close']),2)
#print(price_day1)


@app.route("/")

def index():
    df = inr_df
    actual_chart = go.Scatter(y=df["Close"], x=df.index, name= 'Data')
    

    with TemporaryDirectory() as tmp_dir:
        filename = tmp_dir + "tmp.html"
        py.plot([actual_chart],filename = filename , auto_open=False)
        with open(filename, "r") as f:
            graph_html = f.read()

    
    IS_FORECAST = False
    return render_template("step1.html",price_day1=price_day1, graph_html=graph_html, IS_FORECAST=IS_FORECAST)   


@app.route('/submit',methods=['POST'])
def submit_data():
    try:
        s2=int(request.form['parameter'])
        s1=request.form['options']
    except:
        flash("Please provide valid inputs")
        return redirect("/")

    df = inr_df
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df['Close']).reshape(-1,1))
    ##splitting dataset into train and test split
    training_size=int(len(df1)*0.8)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    print(training_size,test_size)

    fin_train=inr_df[0:training_size-101:1]
    fin_test=inr_df[training_size+101:len(inr_df):1]

    # reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # reshape input to be [samples, time steps, features] which is required for LSTM
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

    if s1=="lstm":
        value="LSTM"
    elif s1=="gru":
        value="GRU"
    elif s1=="arima":
        value="ARIMA" 
    elif s1=="fbprophet":
        value="FBPROPHET"        

    if value == "LSTM":
        seed(100)
        tf.random.set_seed(100) 
        K.clear_session()
        model=Sequential()
        model.add(LSTM(50,input_shape=(100,1),activation='relu',return_sequences=False))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error',optimizer='adam')
        print(model.summary())
        model.fit(X_train,y_train,validation_split=0.25,epochs=100,batch_size=64,verbose=1,shuffle=False)
        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)
        ##Transformback to original form
        print('train_predict',train_predict)
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)

        ### Plotting 
        # shift train predictions for plotting
        look_back=100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
    
        # plot baseline and predictions
        test = pd.DataFrame(scaler.inverse_transform(df1),columns = ["Close"])
        test['trainPredictPlot'] = trainPredictPlot
        test['testPredictPlot'] = testPredictPlot
        data_chart = go.Scatter(y= test['Close'], name= 'Data')
        train_chart = go.Scatter(y= test['trainPredictPlot'], name= 'training data')
        test_chart = go.Scatter(y= test['testPredictPlot'], name= 'testing data')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp3.html"
            py.plot([data_chart,train_chart,test_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph0_html = f.read()

        x_input=test_data[len(test_data)-100:].reshape(1,-1)
        print(x_input.shape)

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        print(temp_input)

        # demonstrate prediction for next days

        lst_output=[]
        n_steps=100
        i=0
        while(i<s2):
    
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                print(x_input)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
    

        print(lst_output)
        day_new=np.arange(1,101)
        day_pred=np.arange(101,101+s2)
        rec = pd.DataFrame(scaler.inverse_transform(df1[len(df1)-100:]),columns = ["Close"])
        rec['Days'] = day_new
        rec2 = pd.DataFrame(scaler.inverse_transform(lst_output),columns = ["Close"])
        rec2['Days'] = day_pred
        dat_chart = go.Scatter(y= rec['Close'],x= rec['Days'], name= 'Data')
        predict_chart = go.Scatter(y= rec2['Close'],x= rec2['Days'], name= 'Predicted')

    
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp.html"
            py.plot([dat_chart,predict_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph_html = f.read()  
        df3_list=df1.tolist()
        df3_list.extend(lst_output)    
        df3 =  pd.DataFrame(scaler.inverse_transform(df3_list).tolist(),columns = ["Close"])
        df4 = df3.tail(s2) 
        data_chart = go.Scatter(y= df3['Close'],x= df3.index, name= 'Original')
        pred_chart = go.Scatter(y= df4['Close'],x= df4.index, name= 'Predicted')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp1.html"
            py.plot([data_chart,pred_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph1_html = f.read()
        final_df_1=rec2[['Close']].head(s2)
        final_df_1.index.names = ['Day']        
        final_df_1.columns = ['Close Predictions']
        final_df_1.index += 1
        IS_FORECAST = True
    
        table = final_df_1.to_html(classes='table table-striped', border=0)
        table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
        table = table.replace('<th></th>', '')
        table = table.replace('<th>', '<th colspan="2">', 1)
        print(table)
        print('Train RMSE: ',math.sqrt(mean_squared_error(fin_train['Close'],train_predict)))
        print('Train MAE: ',mean_absolute_error(fin_train['Close'],train_predict))
        print('Train R2-score:',r2_score(fin_train['Close'],train_predict))
        print('Test RMSE: ',math.sqrt(mean_squared_error(fin_test['Close'],test_predict)))
        print('Test MAE: ',mean_absolute_error(fin_test['Close'],test_predict))
        print('Test R2-score:',r2_score(fin_test['Close'],test_predict))          
        return render_template("step1.html",price_day1=price_day1, graph0_html=graph0_html, graph_html=graph_html,graph1_html=graph1_html,parameter=s2,table=table, IS_FORECAST=IS_FORECAST, IS_LSTM=True)        
    elif value == "GRU":
        seed(200)
        tf.random.set_seed(200)
        K.clear_session()
        model=Sequential()
        model.add(GRU(7,input_shape=(100,1),activation='linear',kernel_initializer='lecun_uniform',return_sequences=False))
        model.add(Dense(1))

        print(model.summary())
        model.compile(loss=tf.keras.metrics.mean_squared_error,metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')],optimizer='adam')
        model.fit(X_train,y_train,epochs=80,validation_split=0.25,batch_size=64,verbose=1,shuffle=False)
        ### Lets Do the prediction and check performance metrics
        train_predict=model.predict(X_train)
        test_predict=model.predict(X_test)
        ##Transformback to original form
        train_predict=scaler.inverse_transform(train_predict)
        test_predict=scaler.inverse_transform(test_predict)

        ### Plotting 
        # shift train predictions for plotting
        look_back=100
        trainPredictPlot = np.empty_like(df1)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(df1)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict

    
        # plot baseline and predictions
        test = pd.DataFrame(scaler.inverse_transform(df1),columns = ["Close"])
        test['trainPredictPlot'] = trainPredictPlot
        test['testPredictPlot'] = testPredictPlot
        data_chart = go.Scatter(y= test['Close'], name= 'Data')
        train_chart = go.Scatter(y= test['trainPredictPlot'], name= 'training data')
        test_chart = go.Scatter(y= test['testPredictPlot'], name= 'testing data')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp3.html"
            py.plot([data_chart,train_chart,test_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph0_html = f.read()

        x_input=test_data[len(test_data)-100:].reshape(1,-1)
        print(x_input.shape)

        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        print(temp_input)

        # demonstrate prediction for next 10 days

        lst_output=[]
        n_steps=100
        i=0
        while(i<s2):
    
            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                #print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model.predict(x_input, verbose=0)
                #print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1
    

        print(lst_output)
        day_new=np.arange(1,101)
        day_pred=np.arange(101,101+s2)
        rec = pd.DataFrame(scaler.inverse_transform(df1[len(df1)-100:]),columns = ["Close"])
        rec['Days'] = day_new
        rec2 = pd.DataFrame(scaler.inverse_transform(lst_output),columns = ["Close"])
        rec2['Days'] = day_pred
        dat_chart = go.Scatter(y= rec['Close'],x= rec['Days'], name= 'Data')
        predict_chart = go.Scatter(y= rec2['Close'],x= rec2['Days'], name= 'Predicted')
    
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp.html"
            py.plot([dat_chart,predict_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph_html = f.read()  
        df3_list=df1.tolist()
        df3_list.extend(lst_output)    
        df3 =  pd.DataFrame(scaler.inverse_transform(df3_list).tolist(),columns = ["Close"])
        df4 = df3.tail(s2) 
        data_chart = go.Scatter(y= df3['Close'],x= df3.index, name= 'Original')
        pred_chart = go.Scatter(y= df4['Close'],x= df4.index, name= 'Predicted')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp1.html"
            py.plot([data_chart,pred_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph1_html = f.read()
        final_df_1=rec2[['Close']].head(s2)
        final_df_1.index.names = ['Day']        
        final_df_1.columns = ['Close Predictions']
        final_df_1.index += 1
        IS_FORECAST = True
    
        table = final_df_1.to_html(classes='table table-striped', border=0)
        table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
        table = table.replace('<th></th>', '')
        table = table.replace('<th>', '<th colspan="2">', 1)
        print(table)
        print('Train RMSE: ',math.sqrt(mean_squared_error(fin_train['Close'],train_predict)))
        print('Train MAE: ',mean_absolute_error(fin_train['Close'],train_predict))
        print('Train R2-score:',r2_score(fin_train['Close'],train_predict))
        print('Test RMSE: ',math.sqrt(mean_squared_error(fin_test['Close'],test_predict)))
        print('Test MAE: ',mean_absolute_error(fin_test['Close'],test_predict))
        print('Test R2-score:',r2_score(fin_test['Close'],test_predict))                   
        return render_template("step1.html",price_day1=price_day1, graph0_html=graph0_html, graph_html=graph_html,graph1_html=graph1_html,parameter=s2,table=table, IS_FORECAST=IS_FORECAST, IS_GRU=True)
    elif value=="ARIMA":
        arima_df = inr_df
        adfuller_test(arima_df['Close'])
        arima_df['Updated Close'] = arima_df['Close'] - arima_df['Close'].shift(1)
        data_chart = go.Scatter(y= arima_df['Updated Close'].dropna(), name= 'Stationary Data')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp.html"
            py.plot([data_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph0_html = f.read()
        print(arima_df.shape)
        train=arima_df.iloc[:-100]
        test=arima_df.iloc[-100:]
        print(train.shape,test.shape) 
        model=sm.tsa.arima.ARIMA(train['Close'],order=(3,0,3))
        model=model.fit()
        print(model.summary())
        
        #Below Calculation for measuring scores of train set
        model3=sm.tsa.arima.ARIMA(arima_df['Close'],order=(3,0,3))
        model3=model3.fit()
        start=0
        end=len(arima_df)-1
        pred_train=model3.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
        pred_train.index = arima_df.index[start:end+1]
        
        #Below Calculation for test set prediction
        start=len(train)
        end=len(train)+len(test)-1
        pred=model.predict(start=start,end=end,typ='levels').rename('ARIMA Predictions')
        pred.index = arima_df.index[start:end+1]
        #pred.plot(legend=True)
        #test['Close'].plot(legend=True)
        dat_chart = go.Scatter(y= pred, name= 'Predicted Test set Data')
        predict_chart = go.Scatter(y= test['Close'], name= 'Original Test set Data')
    
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp1.html"
            py.plot([dat_chart,predict_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph_html = f.read()

        arima2_df = arima_df
        arima2_df=arima2_df.reset_index(drop=True)
        #print("Arima df")
        print(arima2_df)
        model2=sm.tsa.arima.ARIMA(arima2_df['Close'],order=(3,0,3))
        model2=model2.fit()
        print(model2.summary())      

        pred1=model2.predict(start=len(arima2_df),end=len(arima2_df)-1+s2,typ='levels').rename('ARIMA Predictions')
        chart = go.Scatter(y= pred1, name= 'Predicted Data')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp2.html"
            py.plot([chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph1_html = f.read()

        predict = pd.DataFrame(pred1.tolist(),columns=['Close'])
        print(predict)
        final_df = pd.concat([arima2_df, predict],ignore_index=True)
        #print(final_df)
        #print(final_df.tail(20))
        last_val = final_df.tail(s2)
        #print(last_val)
        

        dat_chart = go.Scatter(y= arima2_df['Close'],x=arima2_df.index, name= 'Original')
        predict_chart = go.Scatter(y= last_val['Close'],x=last_val.index, name= 'Predicted')
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp3.html"
            py.plot([dat_chart,predict_chart],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph3_html = f.read()

        final_df_1=predict
        #final_df_1=final_df_1.reset_index(drop=True)
        final_df_1.index.names = ['Day']        
        final_df_1.columns = ['Close Predictions']
        final_df_1.index += 1
        IS_FORECAST = True
    
        table = final_df_1.to_html(classes='table table-striped', border=0)
        table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
        table = table.replace('<th></th>', '')
        table = table.replace('<th>', '<th colspan="2">', 1)
        print(table)      
        print('RMSE: ',math.sqrt(mean_squared_error(arima_df['Close'],pred_train)))
        print('MAE: ',mean_absolute_error(arima_df['Close'],pred_train))
        print('R2-score:',r2_score(arima_df['Close'],pred_train))
        return render_template("step1.html",price_day1=price_day1, graph0_html=graph0_html, graph_html=graph_html, graph1_html=graph1_html, graph3_html=graph3_html, parameter=s2,table=table, IS_FORECAST=IS_FORECAST, IS_ARIMA=True)

    elif value=="FBPROPHET":
        fbpr_df = usdinr_df
        fbpr_df.columns = ['ds','y']

        model = Prophet()
        model.fit(fbpr_df)
        future = model.make_future_dataframe(periods=s2) #MS for monthly, H for hourly
        forecast_data = model.predict(future)
        #forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
        final_df = pd.DataFrame(forecast_data)
        pred_df = final_df.iloc[:len(final_df)-s2]['yhat']

        actual_chart = go.Scatter(y=fbpr_df["y"], name= 'Actual')
        predict_chart = go.Scatter(y=final_df["yhat"], name= 'Predicted')
        predict_chart_upper = go.Scatter(y=final_df["yhat_upper"], name= 'Predicted Upper')
        predict_chart_lower = go.Scatter(y=final_df["yhat_lower"], name= 'Predicted Lower')

        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp.html"
            #print(filename)
            py.plot([actual_chart, predict_chart, predict_chart_upper, predict_chart_lower],filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph0_html = f.read()        
        with TemporaryDirectory() as tmp_dir:
            filename = tmp_dir + "tmp1.html"
            py.plot(plot_components_plotly(model,forecast_data),filename = filename, auto_open=False)
            with open(filename, "r") as f:
                graph4_html = f.read()
        final_df_1=final_df[['yhat']].tail(s2)
        final_df_1=final_df_1.reset_index(drop=True)
        final_df_1.index.names = ['Day']        
        final_df_1.columns = ['Close Predictions']
        final_df_1.index += 1
        IS_FORECAST = True
    
        table = final_df_1.to_html(classes='table table-striped', border=0)
        table = table.replace('tr style="text-align: right;"', 'tr style="text-align: center;"')
        table = table.replace('<th></th>', '')
        table = table.replace('<th>', '<th colspan="2">', 1)
        print(table)
        print('RMSE: ',math.sqrt(mean_squared_error(fbpr_df['y'],pred_df)))
        print('MAE: ',mean_absolute_error(fbpr_df['y'],pred_df))
        print('R2-score:',r2_score(fbpr_df['y'],pred_df))          
        return render_template("step1.html",price_day1=price_day1, graph0_html=graph0_html, graph4_html=graph4_html, parameter=s2,table=table, IS_FORECAST=IS_FORECAST, IS_PROPHET=True)        


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def adfuller_test(close):
    #Ho: It is non stationary
    #H1: It is stationary
    result=adfuller(close)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")     


if __name__ == "__main__":
    app.run(debug = True)
