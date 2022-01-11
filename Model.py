import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv('D:\GIT\Synchronous-Machine-Data-Set\synchronous machine.csv')
LoadCurrent=np.array(df['Iy'].values).reshape(-1,1)
PowerFactor=np.array(df['PF'].values).reshape(-1,1)
PowerFactorError=np.array(df['e'].values).reshape(-1,1)
ChangeInCurrent=np.array(df['dIf'].values).reshape(-1,1)
ExcitationCurrentOfSynchronousMachine=np.array(df['If'].values).reshape(-1,1)
Attributes=np.concatenate((LoadCurrent, PowerFactor, PowerFactorError, ChangeInCurrent), axis=1)

model=LinearRegression()
model.fit(Attributes, ExcitationCurrentOfSynchronousMachine)

Score=model.score(Attributes, ExcitationCurrentOfSynchronousMachine)
print("Accuracy: ", Score*100 , "Percent")

UserLoad=float(input("Enter the Load Current: "))
UserPowerFactor=float(input("Enter the Power Factor: "))
UserPowerFactorError=float(input("Enter the Power Factor Error: "))
UserChangeInCurrent=float(input("Enter the Change In Current: "))

Prediction=model.predict([[UserLoad, UserPowerFactor, UserPowerFactorError, UserChangeInCurrent]])
print("The Excitation Current Of Synchronous Machine is: ", round(Prediction[0][0]), 2)
