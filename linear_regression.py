
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:24:06 2017

@author: abrown09
"""

import pandas as pd
import matplotlib.pyplot as plt

loansData = pd.read_csv('https://github.com/Thinkful-Ed/curric-data-001-data-sets/raw/master/loans/loansData.csv')


# examine data cleanliness
loansData['Interest.Rate'][0:5]
loansData['Loan.Length'][0:5]
loansData['FICO.Range'][0:5]

# need to remove % symbols from Interest.Rate
#cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
#loansData['Interest.Rate'] = cleanInterestRate
cleanInterestRate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')), 4))
loansData['Interest.Rate'] = cleanInterestRate




# need to remove the word months from Loan.Length
cleanLoanLength = loansData['Loan.Length'].map(lambda y: int(y.rstrip(' months')))
loansData['Loan.Length'] = cleanLoanLength

# convert FICO scores into numerics amd save in a new column called 'FICO.score'


#cleanFICORange = loansData['FICO.Range'].map(lambda z: z.split('-'))
#cleanFICORange = cleanFICORange.map(lambda x: [int(n) for n in x])

#loansData['FICO.Range'] = cleanFICORange
#loansData['FICO.Range'][0:5]

loansData['FICO.Score'] = [int(val.split('-')[0]) for val in loansData['FICO.Range']]


plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()

a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')



import numpy as np
import statsmodels.api as sm

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# the dependent variable 
y = np.matrix(intrate).transpose()

#the independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
# why do we do any of this?

""" Interpretation: the p-values of the model are less than .05, which means 
each variable is a significant predictor of the outcome variable, when each of the
other predictors are held constant. 

The R-squared of the model is 0.657, which means the model explains a fair amount of variance
in the outcome.
"""

x = np.column_stack([x1,x2])

# create the linear model:
x = sm.add_constant(x)
model = sm.OLS(y,x)
f = model.fit()

# output the results:
f.summary()

# output dataframe

loansData.to_pickle('loansData_clean.csv')