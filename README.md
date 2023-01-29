# Trading_Stratergy_Prediction_WiDS

This Project is Divided into 3 Part : A)Data Extraction B)Classification of Trade Using XGBoost C)Closing Price Prediction using RNN

A) Data Extraction 
We used API provided by Bitmex to extract hourly candlestick data from 2016 and saved it in the form of CSV

B) Classification of Trade Using XGBoost 
We Prepare the data for further classification and by using various techincal indicatiors. 
We then further use XGBoost to classify the trade and provide importance scores to the various indicators

C) Closing Price Prediction using RNN.
We use the RNN to predict closing price using the data extracted in part A. 
