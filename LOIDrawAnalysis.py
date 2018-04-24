import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = 'C:\\Users\\Andrew\\Desktop\\Data\\'
df = pd.read_csv(path + 'IRL.CSV')

edges = np.arange(0.0,1.05,0.05)
labs = ['0-5%','05-10%','10-15%','15-20%','20-25%','25-30%','30-35%',
        '35-40%','40-45%','45-50%','50-55%','55-60%','60-65%','65-70%',
        '70-75%','75-80%','80-85%','85-90%','90-95%','95-100%']

# Look at all draws

df['DMinProbCat'] = pd.cut(df['MaxD']**-1,bins = edges,labels = labs)

tab = pd.crosstab(df['DMinProbCat'],df['Res'] == 'D')
tab['%'] = tab.ix[:,1]/(tab.ix[:,1]+tab.ix[:,0])
tab = tab.dropna()
tab = tab.reset_index()

#Create bar chart

x = np.arange(len(tab['DMinProbCat']))
h = 0.75 #thickness of bars

plt.style.use('seaborn') # use ggplot theme

plt.barh(x,np.round(tab['%']*100),align = 'center',height = h)
plt.title('League of Ireland 2012 - 2018\n Draw prob. vs rate of occurrence')
plt.yticks(np.arange(len(tab['DMinProbCat'])), tab['DMinProbCat'])
for a,b in zip(np.arange(len(tab['DMinProbCat'])), np.round(tab['%']*100)):
    plt.text(b+0.05, a, str(b))
# Add lines
cutoffs = [10,15,20,25,30,35]
for i in x:
    plt.plot([cutoffs[i]]*2,[-(h/2),i+(h/2)],color = 'red')



# Look at well matched Ds as in book

sub = df[np.absolute(df['AvgH']**-1 - df['AvgA']**-1) < 0.4]

sub['DMinProbCatSub'] = pd.cut(sub['MaxD']**-1,bins = edges,labels = labs)

tab = pd.crosstab(sub['DMinProbCatSub'],sub['Res'] == 'D')
tab['%'] = tab.ix[:,1]/(tab.ix[:,1]+tab.ix[:,0])
tab = tab.dropna()
tab = tab.reset_index()

x = np.arange(len(tab['DMinProbCatSub']))
h = 0.75 #thickness of bars

plt.style.use('ggplot') # use ggplot theme

plt.barh(x,np.round(tab['%']*100),align = 'center',height = h)
plt.title('League of Ireland 2012 - 2018\n Draw prob. vs rate of occurrence')
plt.yticks(np.arange(len(tab['DMinProbCatSub'])), tab['DMinProbCatSub'])
for a,b in zip(np.arange(len(tab['DMinProbCatSub'])), np.round(tab['%']*100)):
    plt.text(b+0.05, a, str(b))
# Add lines
cutoffs = [25,30,35]
for i in x:
    plt.plot([cutoffs[i]]*2,[-(h/2),i+(h/2)],color = 'red')
