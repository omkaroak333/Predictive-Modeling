#############          Final Project          ##############
###### Authors:                                       ######
#              

import os
import pandas as pd
import numpy as np
from datetime import date
import random
import statsmodels.api as sm
import cvxopt as opt
from cvxopt import matrix
from cvxopt import blas,solvers
from sklearn.linear_model import Lasso
from tqdm import tqdm
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from datetime import datetime



#from qpsolvers import solve_qp


#########         Header Code - def of Functions         #########

#Function to compute Info Ratio for given returns
def calcInfoRatio(ReturnArray):
    IR =  np.mean(ReturnArray)/np.std(ReturnArray)
    return IR

#Function to compute Info Ratio for given returns
def calcMaxDrawDown(RETvalues):
    RETvalues = np.array(RETvalues)
    MaxDD = 0
    peakValue = RETvalues[0]
    for ret in RETvalues:
        if ret > peakValue:
            peakValue = ret
        DD = (ret - peakValue) / peakValue
        if DD < MaxDD:
            MaxDD = DD
    return MaxDD

#Funnction to filter out stocks as per part 2
def shortlist_stocks(tempdata):
    #tempdata = tempdata.dropna(subset=['ES'])

    #Filter out top 4000 stocks based on Market-capitalization
    #tempdata = tempdata.sort_values(by=['mcap'],ascending=False)
    #tempdata = tempdata[:4000]

    #Select stocks with ES values falling in top 70 percentile
    #tempdata.ES = tempdata.ES.astype(float)
    #tempdata = tempdata[tempdata.ES >= tempdata.ES.quantile(0.7)]
    startdate = date(2004,12,1)
    enddate = date(2004,12,30)
    tempdata = mergeddata[(mergeddata.DATE >= startdate) & (mergeddata.DATE <= enddate)]
    finallist = set(tempdata['SEDOL']) 
    return finallist


#Function to compute the Covariance Matrix of Returns
def CalCovMatrix(stockdata):

    #Converting data to wide format for easy formulation of covariance
    stackeddata = stockdata[['DATE','SEDOL', 'RETURN']]
    stackeddata = stackeddata.set_index(['DATE','SEDOL'])
    stackeddata = stackeddata.unstack()
    stackeddata.columns = [x[1] for x in stackeddata.columns]
    stackeddata = stackeddata.dropna(axis=1)
    stackeddata = stackeddata.astype(float)
    #Use pandas functions to compute covariance
    covmatrix = stackeddata.cov()
    updatedlist = stackeddata.columns

    #Convert Covariance Matrix to positive-definite matrix
    if np.all(np.linalg.eigvals(covmatrix) > 0):
        covmatrix = covmatrix
    else:
        covmatrix = np.real(sqrtm(covmatrix*covmatrix.T))

    return covmatrix,updatedlist

#Function to compute the Expected Returns based on the given factor
#using Regression Model
def Reg_ExpectedReturns(stockdata,factor,i):
    ##RegResults = pd.DataFrame(index=['const','EP','BP','CP','SP','REP','RBP','RCP','RSP','CTEF','PM1'])
    RegResults = pd.DataFrame(index=['const','EP1','EP2','RV1','RV2','REP','RBP','RCP','RSP','CTEF','LIT'])

    #Run regression for each month in the dataframe
    for d in set(stockdata['DATE']):
        tempdata = stockdata[stockdata['DATE'] == d]

        Y = tempdata['RETURN']
        Y = Y.astype(float)
        ##X = tempdata[['EP','BP','CP','SP','REP','RBP','RCP','RSP','CTEF','PM1']]
        X = tempdata[['EP1','EP2','RV1','RV2','REP','RBP','RCP','RSP','CTEF','LIT']]
	    
        X = X.astype(float)

        if (factor == 'CTEF'):
            X.CTEF = 0

        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        reg = model.fit()

        RegResults[d.strftime('%Y-%m')] = reg.params

    RegResults['Final Coeff'] = RegResults.mean(axis=1)

    #Obtain beta values (average of 24 past months' beta)
    beta = RegResults['Final Coeff']
    ERdata = stockdata[stockdata.DATE == pd.to_datetime(i)]
    ##cols = ['EP','BP','CP','SP','REP','RBP','RCP','RSP','CTEF','PM1']
    cols = ['EP1','EP2','RV1','RV2','REP','RBP','RCP','RSP','CTEF','LIT']
	
    
    
    ERdata = ERdata[['SEDOL'] + cols]
    ERdata['CONST'] = 1
    ERdata[cols] = ERdata[cols].astype(float)

    #Compute the Expected Returns using the beta values and given factors
    ##ERdata['Expected ' + factor] = beta[0]*ERdata.CONST + beta[1]*ERdata.EP + beta[2]*ERdata.BP + beta[3]*ERdata.CP + beta[4]*ERdata.SP + beta[5]*ERdata.REP + beta[6]*ERdata.RBP + beta[7]*ERdata.RCP + beta[8]*ERdata.RSP + beta[9]*ERdata.CTEF + beta[10]*ERdata.PM1
    ERdata['Expected ' + factor] = beta[0]*ERdata.CONST + beta[1]*ERdata.EP1 + beta[2]*ERdata.EP2 + beta[3]*ERdata.RV1 + beta[4]*ERdata.RV2 + beta[5]*ERdata.REP + beta[6]*ERdata.RBP + beta[7]*ERdata.RCP + beta[8]*ERdata.RSP + beta[9]*ERdata.CTEF + beta[10]*ERdata.LIT
	
    
    
    ERdata = ERdata.set_index('SEDOL')
    return ERdata['Expected ' + factor]


#Function to compute the Expected Returns based on the given factor
#using Regression Model
def Reg_ExpectedReturns_newmodel(stockdata,factor,i):
    #RegResults = pd.DataFrame(index=['const','BP','CP','REP','FEP1','CTEF','PM1','FGR1'])
    RegResults = pd.DataFrame(index=['const','EP2','RV1','REP','FEP1','CTEF','LIT','RPM71'])
    
    
    
    
    #Run regression for each month in the dataframe
    for d in set(stockdata['DATE']):
        tempdata = stockdata[stockdata['DATE'] == d]

        Y = tempdata['RETURN']
        Y = Y.astype(float)
        ##X = tempdata[['BP','CP','REP','FEP1','CTEF','PM1','FGR1']]
        X = tempdata[['EP2','RV1','REP','FEP1','CTEF','LIT','RPM71']]
        
        
        
        X = X.astype(float)

        if (factor == 'CTEF'):
            X.CTEF = 0

        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        reg = model.fit()

        RegResults[d.strftime('%Y-%m')] = reg.params

    RegResults['Final Coeff'] = RegResults.mean(axis=1)

    #Obtain beta values (average of 24 past months' beta)
    beta = RegResults['Final Coeff']
    ERdata = stockdata[stockdata.DATE == pd.to_datetime(i)]
    ##cols = ['BP','CP','REP','FEP1','CTEF','PM1','FGR1']
    cols = ['EP2','RV1','REP','FEP1','CTEF','LIT','RPM71']
    
    
    
    ERdata = ERdata[['SEDOL'] + cols]
    ERdata['CONST'] = 1
    ERdata[cols] = ERdata[cols].astype(float)

    #Compute the Expected Returns using the beta values and given factors
    ERdata['Expected ' + factor] = beta[0]*ERdata.CONST + beta[1]*ERdata.EP2 + beta[2]*ERdata.RV1 + beta[3]*ERdata.REP + beta[4]*ERdata.FEP1 + beta[5]*ERdata.CTEF + beta[6]*ERdata.LIT + beta[7]*ERdata.RPM71
    ERdata = ERdata.set_index('SEDOL')
    return ERdata['Expected ' + factor]


def optimal_portfolio(Returns, CovMatrix):

    # Generate mean return vector
    pbar = Returns
    SIGMA = CovMatrix

    numPOS = pbar.size
    varmax = 0.0064

    # Compute A matrix in optimization
    # A is the square root of SIGMA
    U,V = np.linalg.eig(SIGMA)
    # Project onto PSD
    U[U<0] = 0
    Usqrt = np.sqrt(U)
    A = np.dot(np.diag(Usqrt),V.T)

    # Generate G and h matrices
    G1temp = np.zeros((A.shape[0]+1,A.shape[1]))
    G1temp[1:,:] = -A
    h1temp = np.zeros((A.shape[0]+1,1))
    h1temp[0] = np.sqrt(varmax)

    for i in np.arange(numPOS):
        ei = np.zeros((1,numPOS))
        ei[0,i] = 1
        if i == 0:
            G2temp = [matrix(-ei)]
            h2temp = [matrix(np.zeros((1,1)))]
        else:
            G2temp += [matrix(-ei)]
            h2temp += [matrix(np.zeros((1,1)))]

    # Construct list of matrices
    Ftemp = np.ones((1,numPOS))
    F = matrix(Ftemp)
    g = matrix(np.ones((1,1)))

    G = [matrix(G1temp)] + G2temp
    H = [matrix(h1temp)] + h2temp

    # Solve QCQP
    # Passing in -matrix(pbar) since maximizing
    solvers.options['show_progress'] = False
    sol = solvers.socp(-matrix(pbar),Gq=G,hq=H,A=F,b=g)
    xsol = np.array(sol['x'])
    # return answer
    return xsol



#Function to return the returns generated using the Porttfolio Strategy
def PortfolioStrategy_returns(mergeddata,factor):
    FinalPortfolioRET = pd.DataFrame(columns=['RETURN'])

    #Define daterange from Dec-2004 to Nov-2017 to compute returns from Jan-2005 to Dec-2017
    startdate = date(2004,11,30)
    enddate = date(2005,11,1)
    rangeofdates = set(mergeddata.DATE[(mergeddata.DATE >= startdate) & (mergeddata.DATE <= enddate)])
    #Iterate through each date in the daterange
    for i in tqdm(rangeofdates):
        tempdata = mergeddata[mergeddata['DATE'] == i]
        print(i)

        #Call function to shortlist data based on given conditions in part 2
        stocklist = shortlist_stocks(tempdata)
        finaldata = mergeddata[mergeddata.SEDOL.isin(stocklist)]

        #Define startdate of evaluation data 2 year prior to current month-year
        startdate = i.replace(year=i.year-2)
        finaldata = finaldata[(finaldata['DATE'] > startdate) & (finaldata['DATE'] <= i)]

        #Completing the dataframe using forward and then backward fill
        stockdata = pd.DataFrame()
        for SEDOL in set(finaldata['SEDOL']):
            filterdata = finaldata[finaldata.SEDOL == SEDOL]
            filterdata = filterdata.sort_values(by=['DATE'])
            filterdata = filterdata.fillna(method='ffill')
            filterdata = filterdata.fillna(method='bfill')
            stockdata = stockdata.append(filterdata)
        
            
        #Defining the required factors and dropping NAs if present
        ##factors = ['EP','BP','CP','SP','REP','RBP','RCP','RSP','CTEF','PM1','FEP1','FGR1']
        factors = ['EP1','EP2','RV1','RV2','REP','RBP','RCP','RSP','CTEF','LIT']
        
        
        #stockdata = stockdata.dropna(subset=factors)

        #Compute excess return for the given stocksdata
        #stockdata['Excess RET'] = stockdata['RET'] - stockdata['Idx RET']
        stockdata = stockdata.sort_values(by=['SEDOL','DATE'])

        #Call function to generate the Covariance Matrix
        CovMatrix, stocklist = CalCovMatrix(stockdata)
        stockdata = stockdata[stockdata.SEDOL.isin(stocklist)]

        #Calling function to compute Expected Return selectively based on whether it is a ML model or Regression model
        if (factor == 'ML'):
            ExpectedReturn = Reg_ExpectedReturns_newmodel(stockdata,factor,i)
        else:
            ExpectedReturn = Reg_ExpectedReturns(stockdata,factor,i)

        #Call function to generate optimized portfolio weights using Expected return and Covariance matrix computed above
        portfolioweights = pd.DataFrame(optimal_portfolio(ExpectedReturn,CovMatrix), index=ExpectedReturn.index,columns=['weights'])

        #Compute the next month-year combo from the current date in question
        if (i.month == 12):
           try:
               nextdate = i.replace(year=i.year+1,month=1)
           except:
               nextdate = i.replace(month=i.month+1)
        else:
            continue

        #Filter data out for the nextdate and the portfolio stocks
        newdata = mergeddata[(mergeddata.DATE == nextdate) & (mergeddata.SEDOL.isin(portfolioweights.index))]
        newdata = newdata.set_index('SEDOL')
        newdata = newdata['RETURN']

        #Compute final portfolio with optimized weights and returns
        portfolio = pd.concat([portfolioweights,newdata],axis=1).dropna()
        portfolio['weighted RET'] = portfolio['weights']*portfolio['RETURN']
        PortfolioRET = portfolio['weighted RET'].sum()

        #Compute PortfolioReturn for the month-year in question
        final = pd.DataFrame([PortfolioRET],index=[nextdate],columns=['RETURN'])

        #Store Portfolio Return in a dataframe
        FinalPortfolioRET = FinalPortfolioRET.append(final)

    return FinalPortfolioRET


#########             Main Code begins             #########

#Read in Factor Data as provided by the company
#FactorData = pd.read_csv('Total_Data5.csv', low_memory = False)

FactorData = pd.read_csv('C:/Users/TRANSFORMER/Desktop/QCF CLasswork/Computational finance/Final project/rus1000_stocks_factors.csv', 
                         skiprows=4, low_memory = False, names=['Symbol', 'Company Name', 'DATE', 'SEDOL', 'FS_ID', 'RETURN', 'RCP', 'RBP', 'RSP', 'REP', 'RDP', 'RPM71', 'RSTDEV', 'ROE1', 'ROE3', 'ROE5', 'ROA1', 'ROA3', 'ROIC', 'BR1', 'BR2', 'EP1', 'EP2', 'RV1', 'RV2', 'CTEF', '9MFR', '8MFR', 'LIT', 'extra'])
print(FactorData.columns)



rightcol = FactorData.columns[2:29]
wrongcol = FactorData.columns[3:30]
FactorData.loc[FactorData['DATE']==' INC.', rightcol]=FactorData.loc[FactorData['DATE']==' INC.', wrongcol].values
FactorData.loc[FactorData['DATE']==' INC', rightcol]=FactorData.loc[FactorData['DATE']==' INC', wrongcol].values


FactorData = FactorData.drop([0])

#FactorData.to_csv('C:/Users/TRANSFORMER/Desktop/QCF CLasswork/Computational finance/Final project/factordata2.csv')

#print(FactorData)

FactorData['DATE'] = FactorData['DATE'].apply(lambda x: x.strip())

FactorData['DATE'] = FactorData['DATE'].apply(lambda x: str(datetime.strptime(str(x), '%m/%d/%Y').month) 
+ '-' + str(datetime.strptime(str(x), '%m/%d/%Y').day) 
+ '-' + str(datetime.strptime(str(x), '%m/%d/%Y').year))

#FactorData.to_csv('C:/Users/TRANSFORMER/Desktop/QCF CLasswork/Computational finance/Final project/factordata.csv')



type(FactorData['DATE'][1])




#FactorData['DATE'] = pd.to_datetime(FactorData.DATE, format='%Y%m%d', errors='ignore')

#print(FactorData)


#del FactorData['Unnamed: 0']

#Read in Russell 3000 index monthly return data
RUAData = pd.read_csv('C:/Users/TRANSFORMER/Desktop/QCF CLasswork/Computational finance/Final project/Benchmark Returns.csv')
#RUAData['Idx RET'] = (RUAData['Adj Close'] / RUAData['Adj Close'].shift(1))-1
RUAData.rename(columns={'Date':'DATE','Russell 1000 Bench Return':'Idx RET'}, inplace=True)

#print(RUAData)

RUAData['DATE'] = RUAData['DATE'].apply(lambda x: str(datetime.strptime(str(x), '%Y%m%d').month) 
+ '-' + str(datetime.strptime(str(x), '%Y%m%d').day) 
+ '-' + str(datetime.strptime(str(x), '%Y%m%d').year))

#print(RUAData)




#RUAData['DATE'] = pd.to_datetime(RUAData.DATE, format='%Y%m%d', errors='ignore')
#benchmark_data = benchmark_data.rename(columns={'Date':'DATE'})





#RUAData['DATE'] = pd.to_datetime(RUAData.DATE, format='%Y-%m-%d')

#Filter Russell Index Data for required columns
cols = ['DATE','Idx RET']
RUAData = RUAData[cols]

#Convert Return values to numbers from percentages in Factordata
#FactorData['DATE'] = pd.to_datetime(FactorData.DATE, format='%m/%d/%Y')

col_list = list(FactorData.columns)
del col_list[0:5]

#FactorData = FactorData.drop([0])


for col in col_list:
    FactorData[col] = pd.to_numeric(FactorData[col])

FactorData['RETURN'] = FactorData['RETURN']/100


#Merge the two dataframes on Date
mergeddata = FactorData.merge(RUAData, on=['DATE'], how='inner')


type(mergeddata['DATE'][0])

mergeddata['DATE'] = pd.to_datetime(mergeddata.DATE, format='%m-%d-%Y')


#Filter and clean merged data
mergeddata = mergeddata[mergeddata.DATE >= (2004,1,1)]

fctrs = ['EP1','EP2','RV1','RV2','REP','RBP','RCP','RSP','CTEF','LIT']
        
mergeddata = mergeddata.dropna(subset=fctrs)

print(mergeddata)
#dropcols = ['CUSIP','TICKER','GVKey','STATPERS','USFIRM','CURCODE','MRV1','MRV2','TOT']
#mergeddata = mergeddata.drop(columns=dropcols)
mergeddata = mergeddata.sort_values(by=['SEDOL', 'DATE'])
mergeddata = mergeddata.drop_duplicates(subset=['DATE','SEDOL'],keep='first')
mergeddata = mergeddata.replace(to_replace='.',value=np.nan)



mergeddata.to_csv('C:/Users/TRANSFORMER/Desktop/QCF CLasswork/Computational finance/Final project/mergeddatana.csv')

#Define the 3 factors
factor1 = 'RETURN'
factor2 = 'CTEF'
factor3 = 'ML'

#Compute Portfolio Returns using factor = RETURN
#Calculate Info Ratio and Max Drawdown for the portfolio strategy
PortfolioReturns_RET = PortfolioStrategy_returns(mergeddata,factor1)
PortfolioReturns_RET.to_csv('RET Return.csv')
InfoRatio_RET = calcInfoRatio(PortfolioReturns_RET)
Maxdrawdown_RET = calcMaxDrawDown(PortfolioReturns_RET)

#Compute Portfolio Returns using factor = CTEF
#Calculate Info Ratio and Max Drawdown for the portfolio strategy
# PortfolioReturns_CTEF.to_csv('CTEF Return.csv')
# InfoRatio_CTEF = calcInfoRatio(PortfolioReturns_CTEF)
# Maxdrawdown_CTEF = calcMaxDrawDown(PortfolioReturns_CTEF)

#Compute Portfolio Returns using LASSO model
#Calculate Info Ratio and Max Drawdown for the portfolio strategy
PortfolioReturns_Lasso = PortfolioStrategy_returns(mergeddata,factor3)
PortfolioReturns_Lasso.to_csv('ML Return_jugaad.csv')
InfoRatio_Lasso = calcInfoRatio(PortfolioReturns_Lasso)
Maxdrawdown_Lasso = calcMaxDrawDown(PortfolioReturns_Lasso)



