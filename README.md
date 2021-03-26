# CRYPTOCURRENCY CRYSTAL BALL
![logo](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/Header.jpg?raw=true)

## Final Project - Part 1
## Google Slides Link

**Download Presentation** [Goodle Slide Presentation](https://docs.google.com/presentation/d/11-A77KxuFXHH2xDc9bvrYPjjifkujIl8LBE7ybXscUk/edit?usp=sharing)

## Project Overview
For our Columbia University Data Analytics Bootcamp Project our group analyzed cryptocurrencies in two stages to help predict the future price. The first stage was an analysis of the Top 5 most traded coins from 2016-2021, this was to educate both our group and the reader of the financial environment among cryptocurrencies. The second stage was creating a Machine Learning model that would predict these coins prices for the next 30 days.

## Analysis
The data for our Analysis portion of the project came from Facebook Prophet. Facebook Prophet is an opensource software that also provides relatively clean data for cryptocurrencies. The software can help forecast a time series, its built-in models are mainly used to recognize yearly, weekly, and daily seasonality; we used this as an additive to our Machine Learning model. Through FB Prophet and Plotly we were able to create visualizations that helped breakdown the cryptocurrencies market and its activity, like the one below that shows the volume of shares for the top traded coins since 2016. The rest of these visualizations are included in our Google Slides presentation, linked above.

![d1](https://github.com/charlieburd/crypto_crystal_ball/blob/main/Resources/images/coin_volume_breakdown.png)

## Machine Learning Model
To create a machine learning model, we first began by sourcing Bitcoin pricing data from Yahoo Finance Live Data. This data was minute over minute, because the amount of data that needed to be trained, we used the most recent 3 months. Python and Sklearn was used to train/test our cleaned 3-month pricing data. Once the data was trained it was run through a Sklearn SVR (Support Vector Regression) model which was then altered to export a 30-day prediction for Bitcoin pricing

## Database
PostgresSQL was used to store and manipulate data provided by FB Prophet before being imported to Jupyter Notebooks for visualizations.

## Dashboard
Dashboard is hosted by Dash, it shows an interactive graph of the top 12 cryptocurrencies dating from 2016-Present. Our machine learning model is implemented to display a 30-day prediction for the top 5 cryptocurrencies (Bitcoin, Ethereum, Ripple, Litecoin and Tether) based on static data. Multiple coins can be displayed at once and selected areas of the graph can be enlarged with adapting axes. 

![logo](https://github.com/charlieburd/crypto_crystal_ball/blob/main/CrptoCrystalBall_DASHBOARD/assets/Dashboard.png)


# POTENTIAL STOCK PREDICTION

## Final Project - Part 2

## Machine Learning Model - Final Project 2021

In our final project, our team used Machine Learning Support Vector Regression (SVR) model as a Data Science and Analytics cryptocurrency prediction. Bitcoin and other Cryptocurrencies Price Prediction with Machine Learning for the Next 30 Days.

## Team Members - Roles
 * Charlie Burd - Information Manager (Square)
 * Emmanuel Martinez - Tool Creation (Triangle)
 * George Quintanilla - Framework Creation (Circle)

## Resources and DataSets:
This new assignment consists of three technical analysis deliverables and a proposal for further statistical study:

* Data Source: `Bitcoin_1_Min_Historical_Data_2012_to_2020.csv` from [Kaggle - Bitcoin Historical Data](https://www.kaggle.com/mczielinski/bitcoin-historical-data)
* Data Tools:  `ml_yahoo_fianance_stock.py`, `MyBitcoinPredictionApp.ipynb`.
* Software: `CoLab - Google`, `Python 3.9`, `Visual Studio Code 1.50.0`, `Anaconda 4.8.5`, `Jupyter Notebook 6.1.4` and `Pandas` 

## Machine Learning Model (sklearn.svm.SVR)
> Let's have a quick review from Support Vector Regression Model (SVR Model or SVM), below you may find Regression notes and insight information. 

### Support Vector Machines (SVM)

**Support vector machines (SVMs)** are a set of supervised learning methods used for [classification](https://sklearn.org/modules/svm.html#svm-classification), [regression](https://sklearn.org/modules/svm.html#svm-regression) and [outliers detection](https://sklearn.org/modules/svm.html#svm-outlier-detection).

The advantages of support vector machines are:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different [Kernel functions](https://sklearn.org/modules/svm.html#svm-kernels) can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, avoid over-fitting in choosing [Kernel functions](https://sklearn.org/modules/svm.html#svm-kernels) and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see [Scores and probabilities](https://sklearn.org/modules/svm.html#scores-probabilities), below).

The support vector machines in scikit-learn support both dense (`numpy.ndarray` and convertible to that by `numpy.asarray`) and sparse (any `scipy.sparse`) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered `numpy.ndarray` (dense) or `scipy.sparse.csr_matrix` (sparse) with `dtype=float64`.


## Yahoo! Finance as Machine Learning Prediction over the Most US Volatile Stocks 

In our final project, our team inegrated several Jupyter Notebook Modules, Packages and installed multiples Python Libraries to integrate our SVR Module to predict in our 2021 volatile exchange marcket in US. 

Please find below our Machine Learning Solution:

## CORE CODE: (Jupyter Notebook / Python Print Screen Frames)

> Part #1: Packages and Installation.

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%201.JPG?raw=true)

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%202.JPG?raw=true)

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%203.JPG?raw=true)


> Part#2: Import Ipywidgets to Interact Jupyter Notebook with HTML

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%204.JPG?raw=true)


> Testing Interact Package:

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%205.JPG?raw=true)


> Core Code including SVR Model: 

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%206.JPG?raw=true)

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20Part%207.JPG?raw=true)

## MACHINE LEARNING RESULTS:

> Option #1: 15 days Prediction (SVR Model) from DOGE-USD.

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%231.JPG?raw=true)

> Option #1: 30 days Prediction (SVR Model) from DOGE-USD.

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%232.JPG?raw=true)

> Price Date Tange Chart. 

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%233.JPG?raw=true)

> Price Comparison from DOGE-USR vs, a different Crypto:

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%234.JPG?raw=true)

> Past 180 days Stock / Crypto Analysis

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/ML_Yahoo_Finance_Stock%20-%20OPTION%20%235.JPG?raw=true)


### FULL CODE (Jupyter Notebook / Python)

> Below all code used for our Final Project Segment. 

````Python
# **FINAL PROJECT** - Columbia Engineering 2021

# **Potential Stock Prediction**

### **Using nbinteract CLI Usage as #nbi:hide_in to hide our Python Code**

## **PART #1** - Packages Installation.

First, let's install all dependencies to have our Software up to date and with all tools that we need.  

#nbi:hide_in

!pip install nbinteract

#nbi:hide_in

# The next two commands can be skipped for notebook version 5.3 and above

#Currently using version of the notebook server is: 6.1.4

#CODE if you have an older version:
#jupyter nbextension enable --py --sys-prefix bqplot
#jupyter nbextension enable --py --sys-prefix widgetsnbextension

#nbi:hide_in

!pip install sklearn

#nbi:hide_in

!pip install yfinance

#nbi:hide_in

!pip install ipywidgets

#nbi:hide_in

!pip install pandas

#nbi:hide_in

!pip install pandas-datareader

#nbi:hide_in

!pip install numpy

#nbi:hide_in

!pip install plotly

#nbi:hide_in

!pip install datetime

## **PART #2**

## **Import ipywidgets to interact Jupyter Notebook over HTML**

### Key Feature is "Yahoo! Finance" to downloading histrocial market data. 

#nbi:hide_in

from ipywidgets import interact

#nbi:hide_in

import pandas as pd 
from pandas_datareader import data 
import plotly.graph_objects as go 
import plotly.express as px # Plotly Dark Templates
import yfinance as yf # Key Import
from datetime import date 
from dateutil.relativedelta import relativedelta 
import numpy as np 
from sklearn.linear_model import LinearRegression # Machine Leraning (Linear Model)
from sklearn.model_selection import train_test_split # Machine Leraning (Train and Tool Model)

Below the Bash Colors in Python from **python_bash_font** GitHub

#nbi:hide_in

class color:
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   PURPLE = '\033[95m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   END = '\033[0m'

# **PART #3**

# **INTERACT TEST CODE:**

### Using Interact
The ipywidgets library provides the simplest way to get started writing interactive documents. Although the library itself has its own documentation, we will provide a quick overview to let you get started as quickly as possible.

#nbi:hide_in

#Testing Interact from ipywidgets
def square(x):
    return x * x

interact(square, x=10);

#nbi:hide_in

interact(square, x=(0, 100, 10));

#nbi:hide_in

def friends(name, number):
    return '{} has {} friends!'.format(name, number)

#nbi:hide_in

interact(friends, name='George Quinanilla', number=(5, 10));

#nbi:hide_in

interact(friends, name='Charlie Burd', number={'One': 1, 'Five': 5, 'Ten': 10});

## Now we can move on to CORE CODE!

# **CORE CODE:**

As Machine Learning, our SVR (SVM) Model predicts from Yahoo Finance Librery Stock Historcal source. 

From Pandas, "pandas_datareader" we're using "data" library, It's important because help during the process imported every time the END USER Ineraction. 

from pandas_datareader import data


# Date Vars Definicion (YYY-MM-DD)
start_date = '2020-01-01'
end_date = '2020-01-15'


# WHILE-TRUE as END USER Inputs for DataFrame interaction with Yahoo Finance.
# From PANDAS, DATAREADER to get our Final Data interaction.
while True:
    try:
        symbol = input('Enter a valid Stock Tricker Symbol: ')
        # We're using the "symbol" as Stock ticker, then "yahoo" for Yahoo Finance historial data, and last, "start_date" and "end_date" as Date manual input search criteria. 
        df = data.DataReader(symbol, 'yahoo', start_date, end_date)
        break
    # As en error respond: (Example: An invalid Stock Symbol) our END USER gerts the below to submit a valid ticker. 
    except(KeyError, OSError):
        print(color.RED + color.DARKCYAN + f'> {symbol} is not a valid Stock Symbol. Please submit a valid Stock Symbol, Example: BAC for Bank of America' + color.END) 




# Make a User Menu
print(color.BOLD + color.UNDERLINE + color.DARKCYAN + '\n> Please select from below an option to Analyzing your Stock Ticker Data:' + color.END)
choice = True
while choice:
    print(f'''\n SELECT FROM THE BELOW YOUR ANALYSIS OPTION: \n
    1 - Machine Learning 15 days Prediction (SVR Model) from Your Stock "{symbol}" Tricker.
    2 - Machine Learning 30 days Prediction (SVR Model) from Your Stock "{symbol}" Tricker.
    3 - Price Date Range Chart (YYY-MM-DD) from your Stock "{symbol}" Tricker.
    4 - Price Comparison from your Stock "{symbol}" Tricker with another Tricker.
    5 - Your Stock "{symbol}" Tricker 180 Past-Days Analysis.

    Q - Quit and Restart the Process using different Symbol or Stock Tricker.
    ''')
    choice = (input('''\nPlease select a Numeric Option (Example: For the Second Option type "1" and or "Q" to Quit.): '''))
    
    
    
    
    
    
    # ALL CONDITIONS (1 to 5 including Q option)
    
    # OPTION 1: Machine Learning 15 days Prediction from Your Stock "{symbol}" Tricker.
    if choice == '1' or choice == 'A' or choice == 'a':
        
        today = date.today()
        today = today.strftime('%Y-%m-%d')

        df = data.DataReader(symbol, 'yahoo', '2000-01-01', 'today')

        df = df[['Adj Close']]

        # N as Variable for 15 days
        n = 7


        df['Prediction'] = df[['Adj Close']].shift(-n)
        

        # Creating Independent DataSet "X"
        X = df.drop(['Prediction'],1)

        X = np.array(X)

        X = X[:-n]

        # Create the Dependent "Y" DataSet
        Y = df['Prediction']

        Y = np.array(Y)

        # Remove the last "n" rows
        Y = Y[:-n]

        # Split the data into 80% train data and 20 % test data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

        # Create Linear Regression Model
        lr = LinearRegression()
        # Train the model
        lr.fit(x_train, y_train)

        # We want the last 30 rows  
        forecast = np.array(df.drop(['Prediction'],1))[-n:]


        # Print the predictions for the next "n" days
        lr_prediction = lr.predict(forecast)

  
        predictions = pd.DataFrame(lr_prediction, columns = ['Prediction'])
        # "predictions" has 1 column with the predicted values

        df = df.reset_index()

        # From "Date" we need the to get the last value
        d = df['Date'].iloc[-1]
        d = d + relativedelta(days =+ 1)

        # Now we make a list with the respective daterange for our prediction, 15 days after that. 
        datelist = pd.date_range(d, periods = 7).tolist()

        # We add the variable to our Dataframe "predictions"
        predictions['Date'] = datelist
        
        # Save the date of today 3 months older
        trhee_months = date.today() - relativedelta(months=+6)
        trhee_months = trhee_months.strftime('%Y-%m-%d')

        # Get the data for plotting
        df = data.DataReader(symbol, 'yahoo', trhee_months, today)
        df = df.reset_index()

        # Plotting the chart
        fig = go.Figure()
        # Add the data from the first stock
        fig.add_trace(go.Scatter(
                        x=df.Date,
                        y=df['Adj Close'],
                        name=f'{symbol} stock',
                        line_color='dodgerblue',
                        opacity=0.9))
        
        # Add the data from the predictions
        fig.add_trace(go.Scatter(
                        x=predictions.Date,
                        y=predictions['Prediction'],
                        name=f'ML Prediction',
                        line=dict(color='green', dash = 'dot'),
                        opacity=0.9))
    
        fig.update_layout(title=f'Historical {symbol} Quote Stock Value - With 15 Days Prediction',
                                    yaxis_title='Closing Day Price Value in USD',
                                    template='plotly_dark',
                                    xaxis_tickfont_size=14,
                                    yaxis_tickfont_size=14)
        
        fig.show()


   
    
    
    # OPTION 2: Machine Learning 30 days Prediction from Your Stock "{symbol}" Tricker.
    # Same model as above, but using 4 Months data for 30 precition price. 

    elif choice == '2' or choice == 'B' or choice == 'b':
        
        # Get the date of today
        today = date.today()
        # Change the format
        today = today.strftime('%Y-%m-%d')

        df = data.DataReader(symbol, 'yahoo', '2000-01-01', 'today')

        df = df[['Adj Close']]

        # N as Variable for 30 days
        n = 30

        # Create another column "Prediction" shifted "n" units up
        df['Prediction'] = df[['Adj Close']].shift(-n)
        
        X = df.drop(['Prediction'],1)
        X = np.array(X)
        X = X[:-n]


        Y = df['Prediction']
        Y = np.array(Y)
        Y = Y[:-n]

        # Split the data into 80% train data and 20 % test data
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)


        lr = LinearRegression()

        lr.fit(x_train, y_train)
 
        forecast = np.array(df.drop(['Prediction'],1))[-n:]

        lr_prediction = lr.predict(forecast)


        predictions = pd.DataFrame(lr_prediction, columns = ['Prediction'])

        df = df.reset_index()

        d = df['Date'].iloc[-1]
        d = d + relativedelta(days =+ 1)


        datelist = pd.date_range(d, periods = 30).tolist()

        predictions['Date'] = datelist

        

        four_months = date.today() - relativedelta(months=+4)
        four_months = four_months.strftime('%Y-%m-%d')


        df = data.DataReader(symbol, 'yahoo', four_months, today)
        df = df.reset_index()


        fig = go.Figure()

        fig.add_trace(go.Scatter(
                        x=df.Date,
                        y=df['Adj Close'],
                        name=f'{symbol} stock',
                        line_color='dodgerblue',
                        opacity=0.9))
        

        fig.add_trace(go.Scatter(
                        x=predictions.Date,
                        y=predictions['Prediction'],
                        name=f'ML Prediction',
                        line=dict(color='green', dash = 'dot'),
                        opacity=0.9))
    
        fig.update_layout(title=f'Historical {symbol} Quote Stock Value - With 30 Days Prediction',
                                    yaxis_title='Closing Day Price Value in USD',
                                    template='plotly_dark',
                                    xaxis_tickfont_size=14,
                                    yaxis_tickfont_size=14)
        
        fig.show()





    # OPTION 3: Price Date Range Chart from Date Selected. 

    elif choice == '3' or choice == 'C' or choice == 'c':
    
        start_date = input('\nSubmit you Starting Date Analysis (With YYYY-MM-DD Format): ')
        end_date = input('Submit you Ending Date Analysis (With YYYY-MM-DD Format): ')
    
        df = data.DataReader(symbol, 'yahoo', start_date, end_date)
    
        df = df.reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.Date,
                                 y=df['Adj Close'],
                                 line_color='dodgerblue'))
        
        fig.update_layout(title=f'{symbol} Stock Price from {start_date} to {end_date}',
                                    yaxis_title='Closing Day Price Value in USD',
                                    template='plotly_dark',
                                    xaxis_tickfont_size=14,
                                    yaxis_tickfont_size=14)

        fig.show()
     
    
    



    # OPTION 4: Price Comparison from your Stock Tricker.
    elif choice == '4' or choice == 'D' or choice == 'd':
        start_date = input('\nSubmit you Starting Date Analysis (With YYYY-MM-DD Format): ')
        end_date = input('Submit you Ending Date Analysis (With YYYY-MM-DD Format): ')
    
        df = data.DataReader(symbol, 'yahoo', start_date, end_date)
        df = df.reset_index()
        
        while True:
            try:
                symbol_2 = input(f'\nWith which stock would you like to compare {symbol} stock? \nPlease enter a valid Stock Symbol: ')
                df_2 = data.DataReader(symbol_2, 'yahoo', start_date, end_date)
                df_2 = df_2.reset_index()
                break
            except(KeyError, OSError):
                print(color.BOLD + color.UNDERLINE + f'> {symbol_2} is not a valid stock symbol. Please try again...' + color.END)

                
        fig = go.Figure()

        fig.add_trace(go.Scatter(
                    x=df.Date,
                    y=df['Adj Close'],
                    name=f'{symbol} Stock',
                    line_color='dodgerblue',
                    opacity=0.9))
        
        fig.add_trace(go.Scatter(
                    x=df_2.Date,
                    y=df_2['Adj Close'],
                    name=f'{symbol_2} Stock',
                    line_color='dimgray',
                    opacity=0.9))
    
    
        fig.update_layout(title=f'Price Comparison of {symbol} Stock and {symbol_2} Stock from {start_date} to {end_date}', 
                                    yaxis_title='Closing Day Price Value in USD',
                                    template='plotly_dark',
                                    xaxis_tickfont_size=14,
                                    yaxis_tickfont_size=14)
        
        fig.show()
    

        
   
    
    
    # Option 5: Your Stock Tricker 180 Past-Days Analysis.
    elif choice == '5' or choice == 'E' or choice == 'e':

        try:
            # Save the date of today in the variable "today"
            today = date.today()
            # We convert the type of the variable in the format %Y-%m-%d
            today = today.strftime('%Y-%m-%d')
            # Save the date of today 6 months ago, by subtracting 6 months from the date of today
            six_months = date.today() - relativedelta(months=+6)
            six_months = six_months.strftime('%Y-%m-%d')
        
            df2 = yf.Ticker(symbol)
            # Save the Analyst Recommendations in "rec"
            rec = df2.recommendations
            # The DataFrame "rec" has 4 columns: "Firm", "To Grade", "From Grade" and "Action"
            # The index is the date ("DatetimeIndex")
    
            # Now we select only those columns which have the index(date) from "six months" to "today"
            rec = rec.loc[six_months:today,]
        
            # Unfortunately in some cases no data is available, so that the DataFrame is empty. Then the user gets the following message
            if rec.empty:
                print(color.BOLD + color.UNDERLINE + "\n> Unfortunately, there are no recommendations by analysts provided for your chosen stock!" + color.END)
                    
                    
            else:    
                # Replace the index with simple sequential numbers and save the old index ("DatetimeIndex") as a variable "Date"
                rec = rec.reset_index()
    
                # For our analysis we don't need the variables/columns "Firm", "From Grade" and "Action", therefore we delete them
                rec.drop(['Firm', 'From Grade', 'Action'], axis=1, inplace=True)

                # We change the name of the variables/columns
                rec.columns = (['date', 'grade'])
        
                # Now we add a new variable/column "value", which we give the value 1 for each row in order to sum up the values based on the contents of "grade"
                rec['value'] = 1

                # Now we group by the content of "grade" and sum their respective values 
                rec = rec.groupby(['grade']).sum()
                # The DataFrame "rec" has now 1 variable/column which is the value, the index are the different names from the variable "grade"
                # However for the plotting we need the index as a variable 
                rec = rec.reset_index()
        
                # For the labels we assign the content/names of the variable "grade" and for the values we assign the content of "values" 
                fig = go.Figure(data=[go.Pie(labels=rec.grade,
                                                values=rec.value,
                                                hole=.3)])
                # Give a title
                fig.update_layout(template='plotly_dark', title_text=f'Analyst Recommendations of {symbol} Stock from {six_months} to {today}')

                # Plotting the chart
                fig.show()  
            

    
        # For some stocks the imported data is distorted and in a wrong format, so that an error appears
        # In this cases the user gets the following message:
        except(ValueError,AttributeError):
            print(color.BOLD + color.UNDERLINE + '\n> Unfortunately, there are no recommendations provided for your chosen stock!' + color.END) 

 

    
    # OPTION 6: Quit the program
    elif choice == '6' or choice == 'Q' or choice == 'q':
        print(color.BLUE + color.BOLD + '\n> If you like our results, please share our solution with others! Thank you!' + color.END)
        choice = None
        

    # If user inputs a non valid option
    else:
        print(color.BOLD + color.UNDERLINE + '\n> Your Selection is INVALID! Please select a correct Option from our list.' + color.END)



**Potential Stock Prediction**
```` 


## DOCUMENTATION
### Regression

The method of Support Vector Classification can be extended to solve regression problems. This method is called Support Vector Regression.

The model produced by support vector classification (as described above) depends only on a subset of the training data, because the cost function for building the model does not care about training points that lie beyond the margin. Analogously, the model produced by Support Vector Regression depends only on a subset of the training data, because the cost function for building the model ignores any training data close to the model prediction.

There are three different implementations of Support Vector Regression: `SVR`, `NuSVR` and `LinearSVR`. `LinearSVR` provides a faster implementation than `SVR` but only considers linear kernels, while `NuSVR` implements a slightly different formulation than `SVR` and `LinearSVR`. See Implementation details for further details.

As with classification classes, the fit method will take as argument vectors X, y, only that in this case y is expected to have floating point values instead of integer values:

````Python
>>> from sklearn import svm
>>> X = [[0, 0], [2, 2]]
>>> y = [0.5, 2.5]
>>> clf = svm.SVR()
>>> clf.fit(X, y) 
SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
    kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
>>> clf.predict([[1, 1]])
array([ 1.5])
```` 
Examples:
[Support Vector Regression (SVR) using linear and non-linear kernels](https://sklearn.org/auto_examples/svm/plot_svm_regression.html#sphx-glr-auto-examples-svm-plot-svm-regression-py)

### Mathematical Formulation

A support vector machine constructs a hyper-plane or set of hyper-planes in a high or infinite dimensional space, which can be used for classification, regression or other tasks. Intuitively, a good separation is achieved by the hyper-plane that has the largest distance to the nearest training data points of any class (so-called functional margin), since in general the larger the margin the lower the generalization error of the classifier.

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/mf1.png?raw=true)



### Classification
`SVC`, `NuSVC` and `LinearSVC` are classes capable of performing multi-class classification on a dataset.

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/SVC_Class.png?raw=true)


`SVC` and `NuSVC` are similar methods, but accept slightly different sets of parameters and have different mathematical formulations (see section Mathematical formulation). On the other hand, `LinearSVC` is another implementation of Support Vector Classification for the case of a linear kernel. Note that `LinearSVC` does not accept keyword `kernel`, as this is assumed to be linear. It also lacks some of the members of `SVC` and `NuSVC`, like `support_`.

As other classifiers, `SVC`, `NuSVC` and `LinearSVC` take as input two arrays: an array X of size `[n_samples, n_features]` holding the training samples, and an array y of class labels (strings or integers), size `[n_samples]`:

````Python
>>> from sklearn import svm
>>> X = [[0, 0], [1, 1]]
>>> y = [0, 1]
>>> clf = svm.SVC()
>>> clf.fit(X, y)  
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
````

After being fitted, the model can then be used to predict new values:

````Python
>>> clf.predict([[2., 2.]])
array([1])
````

SVMs decision function depends on some subset of the training data, called the support vectors. Some properties of these support vectors can be found in members `support_vectors_`, `support_` and `n_support`:

````Python
>>> # get support vectors
>>> clf.support_vectors_
array([[ 0.,  0.],
       [ 1.,  1.]])
>>> # get indices of support vectors
>>> clf.support_ 
array([0, 1]...)
>>> # get number of support vectors for each class
>>> clf.n_support_ 
array([1, 1]...)
````

### Multi-Class Classification¶
`SVC` and `NuSVC` implement the “one-against-one” approach (Knerr et al., 1990) for multi- class classification. If `n_class` is the number of classes, then `n_class * (n_class - 1) / 2` classifiers are constructed and each one trains data from two classes. To provide a consistent interface with other classifiers, the `decision_function_shape` option allows to aggregate the results of the “one-against-one” classifiers to a decision function of shape `(n_samples, n_classes)`:

````Python
>>> X = [[0], [1], [2], [3]]
>>> Y = [0, 1, 2, 3]
>>> clf = svm.SVC(decision_function_shape='ovo')
>>> clf.fit(X, Y) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes: 4*3/2 = 6
6
>>> clf.decision_function_shape = "ovr"
>>> dec = clf.decision_function([[1]])
>>> dec.shape[1] # 4 classes
4
````

On the other hand, `LinearSVC` implements “one-vs-the-rest” multi-class strategy, thus training n_class models. If there are only two classes, only one model is trained:

````Python
>>> lin_clf = svm.LinearSVC()
>>> lin_clf.fit(X, Y) 
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
>>> dec = lin_clf.decision_function([[1]])
>>> dec.shape[1]
4
````

See [Mathematical formulation](https://sklearn.org/modules/svm.html#svm-mathematical-formulation) for a complete description of the decision function.

Note that the LinearSVC also implements an alternative multi-class strategy, the so-called multi-class SVM formulated by Crammer and Singer, by using the option multi_class='crammer_singer'. This method is consistent, which is not true for one-vs-rest classification. In practice, one-vs-rest classification is usually preferred, since the results are mostly similar, but the runtime is significantly less.


### Theming and templates in Python

The Plotly Python library comes pre-loaded with several themes that you can get started using right away, and it also provides support for creating and registering your own themes.

> Note on terminology: Theming generally refers to the process of defining default styles for visual elements. Themes in plotly are implemented using objects called templates. Templates are slightly more general than traditional themes because in addition to defining default styles, templates can pre-populate a figure with visual elements like annotations, shapes, images, and more. In the documentation we will refer to the overall process of defining default styles as theming, and when in comes to the plotly API we will talk about how themes are implemented using templates.

### Using built-in themes
##### View available themes
To see information about the available themes and the current default theme, display the plotly.io.templates configuration object like this.

````python
import plotly.io as pio
pio.templates
````

Templates configuration
-----------------------
````python
    Default template: 'plotly'
    Available templates:
        ['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']
````

From this, we can see that the default theme is "plotly", and we can see the names of several additional themes that we can choose from.

### Specifying themes in Plotly Express
All Plotly Express functions accept a template argument that can be set to the name of a registered theme (or to a Template object as discussed later in this section). Here is an example of using Plotly Express to build and display the same scatter plot with six different themes.

````python
import plotly.express as px

df = px.data.gapminder()
df_2007 = df.query("year==2007")

for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
    fig = px.scatter(df_2007,
                     x="gdpPercap", y="lifeExp", size="pop", color="continent",
                     log_x=True, size_max=60,
                     template=template, title="Gapminder 2007: '%s' theme" % template)
    fig.show()
````


![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/pgl1.JPG?raw=true)

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/pgl2.JPG?raw=true)

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/pgl3.JPG?raw=true)



### Specifying themes in graph object figures
The theme for a particular graph object figure can be specified by setting the template property of the figure's layout to the name of a registered theme (or to a Template object as discussed later in this section). Here is an example of constructing a surface plot and then displaying it with each of six themes.

````python
import plotly.graph_objects as go
import pandas as pd

z_data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv")

fig = go.Figure(
    data=go.Surface(z=z_data.values),
    layout=go.Layout(
        title="Mt Bruno Elevation",
        width=500,
        height=500,
    ))

for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
    fig.update_layout(template=template, title="Mt Bruno Elevation: '%s' theme" % template)
    fig.show()
    
````


![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/pgl9.JPG?raw=true)

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/pgl10.JPG?raw=true)


### Specifying a default themes
If a theme is not provided to a Plotly Express function or to a graph object figure, then the default theme is used. The default theme starts out as "plotly", but it can be changed by setting the plotly.io.templates.default property to the name of a registered theme.

Here is an example of changing to default theme to "plotly_white" and then constructing a scatter plot with Plotly Express without providing a template.

Note: Default themes persist for the duration of a single session, but they do not persist across sessions. If you are working in an IPython kernel, this means that default themes will persist for the life of the kernel, but they will not persist across kernel restarts.

````python
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"

df = px.data.gapminder()
df_2007 = df.query("year==2007")

fig = px.scatter(df_2007,
                 x="gdpPercap", y="lifeExp", size="pop", color="continent",
                 log_x=True, size_max=60,
                 title="Gapminder 2007: current default theme")
fig.show()
````

![d1](https://github.com/emmanuelmartinezs/PotentialStockPrediction/blob/main/Resources/Images/pgl15.JPG?raw=true)


### Disable default theming
If you do not wish to use any of the new themes by default, or you want your figures to look exactly the way they did prior to plotly.py version 4, you can disable default theming by setting the default theme to "none".

````python
import plotly.io as pio
pio.templates.default = "none"
````

#### Reference:  [Plotly | Graphing Libraries (Fundamentals and Theming Templates)](https://plotly.com/python/templates/)


#### By Emmanuel Martinez, George Quintanilla and Charlie Burd.
> FINAL PROJECT - Columbia Engineering 2021
