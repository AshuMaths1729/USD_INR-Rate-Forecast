def forecast():
    import pandas as pd
    from sklearn import linear_model
    import datetime as dt
    import matplotlib.pyplot as plt
    import quandl
    # data = quandl.get("FED/RXI_N_B_IN")
    data = quandl.get_table('FED/RXI_N_B_IN', paginate=True, qopts={'columns': ['date', 'rate']})
    rates = {}
    rates = {'date':[x for x in data.Date],
             'rate':[y for y in data.Value]}
    df = pd.DataFrame(rates, columns = ["date", "rate"])
    dates = df['date']
    rates = df['rate']
    dates = pd.to_datetime(dates)
    dates = dates.map(dt.datetime.toordinal)
    
    test_size = int(0.2*dates.size)
    train_size = dates.size - int(0.2*dates.size)
    #test data
    dates_test = dates[0:test_size].values.reshape(test_size,1)
    rates_test = rates[0:test_size].values.reshape(test_size,1)
    
    #train data
    dates_train = dates[test_size:dates.size].values.reshape(train_size,1)
    rates_train = rates[test_size:dates.size].values.reshape(train_size,1)
    
    plt.plot_date(dates_train, rates_train, fmt="g-")
    plt.title('US-INR Currency Predictor [TRAIN]')
    plt.ylabel("INR rate against $1")
    plt.xlabel("Date")
    
    #training
    regr = linear_model.LinearRegression()
    regr.fit(dates_train,rates_train)
    rates_pred = regr.predict(dates_test)
    plt.scatter(dates_test, rates_test,  color='black', linewidth= 1)
    plt.title('Data')
    plt.xlabel('Date')
    plt.ylabel('Rate')
    plt.plot(dates_test, rates_pred, color = 'red', linewidth= 3)
    plt.xticks(())
    plt.yticks(())
    
    y,m,d = map(int, input("Enter date in YYYY MM DD format: ").split())
    date = dateToOrdinal(y,m,d)
    print( str((regr.predict(date))))  


#Converting date to ordinals
def dateToOrdinal(y,m,d):
   import datetime
   d= datetime.date(y,m,d)
   return d.toordinal()