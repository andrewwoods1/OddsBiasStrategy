import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import sklearn as skl


# Replicate the odds bias strategy seen in Soccermatics
# for the LOI.

path = 'C:\\Users\\Andrew\\Desktop\\Data\\'

df = pd.read_csv(path + 'IRL.CSV')

df.head()
df.describe()
#1245 games

df['Home'].value_counts()
df['Away'].value_counts()

sum_of_probs = (df['MaxH']**-1) + (df['MaxD']**-1) + (df['MaxA']**-1)
np.sum(sum_of_probs < 1)/float(len(sum_of_probs))
#Opportunities for arbitrage existed on 50% of the past 1245 LOI games


#Bin the odds and look at success records in the data set

edges = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]
labs = ['0-5%','05-10%','10-15%','15-20%','20-25%','25-30%','30-35%','35-40%',
                              '40-45%','45-50%','50-55%','55-60%','60-65%','65-70%','70-75%','75-80%','80-85%','85-90%','90-95%','95-100%']

#Home wins

df['MinProbH'] = (df['MaxH']**-1)
df['HomeMaxCat'] = pd.cut(df['MinProbH'],bins = edges,
                    labels = labs)
Res = df['Res'] == 'H'
tab = pd.crosstab(df['HomeMaxCat'],Res)
tab['%'] = (tab.ix[:,1]*100)/(tab.ix[:,1] + tab.ix[:,0])
tab['Over'] = tab['%'] - edges[1:]
tab = tab.sort_values(by = ['%'])

plt.style.use('seaborn')
plt.barh(y = tab.index,width = tab.ix[:,2],height = 0.75)
plt.xlabel('Rate of Occurrence %')
plt.ylabel('Bookmaker Probability %')
plt.title('Prob. implied by odds vs rate of occurrence')
plt.suptitle('League of Ireland Home Wins - 2012 - 2018')
for a,b in zip(tab.ix[:,2], tab.index):
    plt.text(a+0.008, b, str(np.round(a,2)),verticalalignment='center')


# Simulate betting on all home wins which have odds of in lower 'value' range (<=1.25)

bank = 1000
df_h = df[df['MaxH']<=1.25]
df_h = df_h.reset_index()
bal = list()
res = df_h['Res']

for i in range(len(df_h)):
    stake = bank*0.1
    if res[i] == 'H':
        bank = bank + (df_h['MaxH'].iloc[i]-1)*stake
    else:
        bank = bank - stake
    bal.append(bank)

plt.plot(range(len(bal)),bal,'--')
plt.title("Bank vs Bet No.")
plt.xlabel("Bet")
plt.ylabel("Balance")


#Create logistic model to map biased odds to true ones

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df['HomeWin'] = df['Res']== 'H'
df['Int'] = 1


X_train, X_test, y_train, y_test = train_test_split(df[['Int','AvgH','AvgD','AvgA','MaxH','Res']],
                                                        df['HomeWin'], test_size=0.5, random_state = 42)


# Below code uses 2017 and onward as test data rather than random allocation as above
#x_train = df[['Int','AvgH','AvgD','AvgA']].iloc[0:991,]
#y_train = df['HomeWin'].iloc[0:991,]
#
#x_test = df[['Int','AvgH','AvgD','AvgA']].iloc[992:,]
#y_test = df['HomeWin'].iloc[992:,]




logreg = LogisticRegression()
logreg.fit(X_train[['Int','AvgH','AvgD','AvgA']],y_train)

probs = logreg.predict_proba(X_test[['Int','AvgH','AvgD','AvgA']])
plt.plot(probs,(df['MaxH'].ix[X_test.index])**-1,'.')
plt.title('Fitted Probabilities vs Max available odds probabilities')
plt.xlabel('Fitted Probabilities')
plt.ylabel('Odds Probabilities')
plt.plot([0, 1], [0, 1], 'k-', color = 'r')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()



# Test whether betting according to logistic model would have made a profit

X_test_copy = X_test.reset_index()
bank = 1000
bal = list()
stakes = list()
kellys = list()
returns = list()

for i in range(len(probs)):
    if (probs[i][1] > X_test_copy['MaxH'][i]**-1) & (X_test_copy['MaxH'][i] < 1.25):
        bal.append(bank)
        kelly = (((X_test_copy['MaxH'][i] - 1) * (probs[i][1]) - (1 - probs[i][1]))/float(X_test_copy['MaxH'][i] - 1))*0.2

        if kelly < 1:
            stake = kelly*bank
            stakes.append(stake)
            kellys.append(kelly)
            if X_test_copy['Res'][i] == 'H':
                bank = bank + (X_test_copy['MaxH'][i] - 1)*stake
            else:
                bank = bank - stake
        else:
            stake = 0.1 * bank
            stakes.append(stake)
            kellys.append(0.1)
            if X_test_copy['Res'][i] == 'H':
                bank = bank + (X_test_copy['MaxH'][i] - 1) * stake
            else:
                bank = bank - stake

    else:
        bank = bank
returns.append(float(bank - 1000)/1000)



#################################


#################################
# Run for multiple test sets in a loop and calculate average returns

returns = list()
for rand in np.random.randint(1,1000,500):

    df['HomeWin'] = df['Res']== 'H'
    df['Int'] = 1


    X_train, X_test, y_train, y_test = train_test_split(df[['Int','AvgH','AvgD','AvgA','MaxH','Res']],
                                                            df['HomeWin'], test_size=0.5, random_state = rand)

    logreg = LogisticRegression()
    logreg.fit(X_train[['Int','AvgH','AvgD','AvgA']],y_train)

    X_test_copy = X_test.reset_index()
    bank = 1000
    bal = list()
    stakes = list()
    kellys = list()

    for i in range(len(probs)):
        if (probs[i][1] > X_test_copy['MaxH'][i]**-1): #& (X_test_copy['MaxH'][i] < 1.25):
            bal.append(bank)
            kelly = (((X_test_copy['MaxH'][i] - 1) * (probs[i][1]) - (1 - probs[i][1]))/float(X_test_copy['MaxH'][i] - 1))*0.5

            if kelly < 1: # Defunct code that can be used to cap the kelly proportion
                stake = kelly*bank
                stakes.append(stake)
                kellys.append(kelly)
                if X_test_copy['Res'][i] == 'H':
                    bank = bank + (X_test_copy['MaxH'][i] - 1)*stake
                else:
                    bank = bank - stake
            else:
                stake = 0.1 * bank
                stakes.append(stake)
                kellys.append(0.1)
                if X_test_copy['Res'][i] == 'H':
                    bank = bank + (X_test_copy['MaxH'][i] - 1) * stake
                else:
                    bank = bank - stake

        else:
            bank = bank
    returns.append(float(bank - 1000)/1000)








