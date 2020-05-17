import numpy
import math
import pickle
import pandas
import FundamentalAnalysis
    
def selectIndustry():
    companies = FundamentalAnalysis.available_companies()
    companiesNasdaqGlobalSelect=companies[companies['exchange']=='Nasdaq Global Select']
    print(companiesNasdaqGlobalSelect)
    technologyTickers=[]

    for ticker in companiesNasdaqGlobalSelect.index.tolist():
        profile = FundamentalAnalysis.profile(ticker)
        if profile.loc['sector','profile']=='Technology' and float(profile.loc['mktCap','profile'])>7e9:
            technologyTickers.append(ticker)
            print(ticker)
            print(float(profile.loc['mktCap','profile']))
            print("")

    filename = 'technologyTickers'
    outfile = open(filename,'wb')
    pickle.dump(technologyTickers,outfile)
    outfile.close()
 
 
def getSavedData():
    filename = 'technologyTickers'
    infile = open(filename,'rb')
    technologyTickers = pickle.load(infile, encoding='latin1')
    return technologyTickers
    
def dataReader(ticker = "AAPL"):

    # ticker = "AAPL"
    # selectIndustry()
    # print(getSavedData())
    
    growth_quarterly = FundamentalAnalysis.financial_statement_growth(ticker, period="quarter").T
    for (columnName, columnData) in growth_quarterly.iteritems():
        growth_quarterly[columnName] = pandas.to_numeric(columnData, errors='coerce').fillna(0) 
        # growth_quarterly[columnName] =  growth_quarterly[columnName]      
    # print(growth_quarterly)  
    # growth_quarterly.to_csv('growth_quarterly.csv')
    tickerMetrics=growth_quarterly
    
    
    balance_sheet_quarterly = FundamentalAnalysis.balance_sheet_statement(ticker, period="quarter").T
    for (columnName, columnData) in balance_sheet_quarterly.iteritems():
        balance_sheet_quarterly[columnName] = pandas.to_numeric(columnData, errors='coerce').fillna(0) *1e-11 
    tickerMetrics=tickerMetrics.join(balance_sheet_quarterly)
    # print(balance_sheet_quarterly)
    # balance_sheet_quarterly.to_csv('balance_sheet_quarterly.csv')
    
    
    key_metrics_quarterly = FundamentalAnalysis.key_metrics(ticker, period="quarter").T
    for (columnName, columnData) in key_metrics_quarterly.iteritems():
        key_metrics_quarterly[columnName] = pandas.to_numeric(columnData, errors='coerce').fillna(0)*1e-2
    key_metrics_quarterly['Market Cap'] = key_metrics_quarterly['Market Cap']*1e-10
    key_metrics_quarterly['Enterprise Value'] = key_metrics_quarterly['Enterprise Value']*1e-10
    key_metrics_quarterly['Working Capital'] = key_metrics_quarterly['Working Capital']*1e-10
    key_metrics_quarterly['Tangible Asset Value'] = key_metrics_quarterly['Tangible Asset Value']*1e-10
    key_metrics_quarterly['Net Current Asset Value'] = key_metrics_quarterly['Net Current Asset Value']*1e-10
    key_metrics_quarterly['Invested Capital'] = key_metrics_quarterly['Invested Capital']*1e-10
    key_metrics_quarterly['Average Receivables'] = key_metrics_quarterly['Average Receivables']*1e-10
    key_metrics_quarterly['Average Payables'] = key_metrics_quarterly['Average Payables']*1e-10
    key_metrics_quarterly['Average Inventory'] = key_metrics_quarterly['Average Inventory']*1e-10
    
    tickerMetrics=tickerMetrics.join(key_metrics_quarterly)
    # print(key_metrics_quarterly)
    # key_metrics_quarterly.to_csv('key_metrics_quarterly.csv')

    tickerMetrics.index = pandas.to_datetime(tickerMetrics.index)
    tickerMetrics=tickerMetrics.loc[tickerMetrics.index > '2009-5-01 00:00:00']
    
    # print (tickerMetrics)
    stock_data_detailed = FundamentalAnalysis.stock_data_detailed(ticker, begin="2010-01-01", end="2020-01-01")
    del stock_data_detailed['label']
    del stock_data_detailed['unadjustedVolume']
    for (columnName, columnData) in stock_data_detailed.iteritems():
        stock_data_detailed[columnName] = pandas.to_numeric(columnData, errors='coerce').fillna(0) 
    stock_data_detailed['volume'] = stock_data_detailed['volume']*1e-8
    stock_data_detailed.insert(0, 'index',  range(0, len(stock_data_detailed)))
    stock_data_detailed.index = pandas.to_datetime(stock_data_detailed.index)

    # print (stock_data_detailed)
    return stock_data_detailed,tickerMetrics

if __name__ == '__main__':
    # dataReader()
    print(getSavedData())