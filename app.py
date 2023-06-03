from flask import Flask, request, render_template
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

def generate_plot(x):
    print("entered")
    plt.figure(figsize=(15,4))
    plt.plot(x,label='predicted data')
    plt.legend()
    plt.grid(), plt.margins(x=0)
    plt.xlabel("Date")
    plt.ylabel("Consumption in KW")
    graph_filename = 'static/graph.png'
    plt.savefig(graph_filename)



data_daily=pd.read_csv('data_daily.csv',low_memory=False)
data_daily['time'] = pd.to_datetime(data_daily['time'])
data_daily.set_index('time', inplace=True)
n = 1
X = data_daily.values
size = int(len(X) * 0.7)
train_arima, test_arima = X[0:size], X[size:len(X)]


def predict(date_predict,no_days):
 predictions_future=list()
 confidence_predict=list()
 history = [x for x in train_arima]
 start = datetime.datetime.strptime(date_predict, "%Y-%m-%d")
 date_list = [start + relativedelta(days=x) for x in range(0,no_days)]
 future_prediction = pd.DataFrame(index=date_list, columns= data_daily.columns)
 data_predict=future_prediction[['Overall_usage']].copy()
 for t in range(0,len(data_predict),n):
     model = ARIMA(history, order=(2,0,5))
     model_fit = model.fit()
     output = model_fit.forecast(n).tolist()
     conf = model_fit.get_forecast(n).conf_int(0.05)
     predictions_future.extend(output)
     confidence_predict.extend(conf)
     obs = test_arima.tolist()[t:t+n]
     history = history[n:]
     history.extend(obs);  
 data_predict['Overall_usage']=predictions_future
 generate_plot(data_predict['Overall_usage'])
 print("ended") 
 return data_predict


app = Flask(__name__)

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/', methods=['GET', 'POST'])
def home():
   
        
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def index():
    from_date = request.form['from_date']
    no_of_days = request.form['no_of_days']
    print(from_date)
    predicted_data=predict(from_date,int(no_of_days))
    
    return render_template('index.html', plot_filename='static/graph.png',table=predicted_data.to_html())



if __name__ == '__main__':
    app.run(debug=True)