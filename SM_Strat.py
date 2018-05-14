import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path = "C:\Downloads\\IRL.csv"
df = pd.read_csv(path)


bank = 1000
bal = [bank]
for i in range(len(df)):

    if df['AvgH'][i]**-1 - df['AvgA'][i]**-1 > 0.4:
        stake = bank * 0.1
        if df['Res'][i] == 'H':
            bank = bank + (df['MaxH'][i] - 1)*stake
        else:
            bank = bank - stake
    elif df['AvgH'][i]**-1 - df['AvgA'][i]**-1 < 0.15:
        stake = bank * 0.1
        if df['Res'][i] == 'D':
            bank = bank + (df['MaxD'][i] - 1)*stake
        else:
            bank = bank - stake
    else:
        bank = bank
    bal.append(bank)


plt.plot(range(len(bal)),bal,'--')
plt.title("Bank vs Bet No.")
plt.xlabel("Bet")
plt.ylabel("Balance")
