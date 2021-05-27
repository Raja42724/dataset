import pandas
db =pandas.read_csv("Salary_Data.csv")
db.info
db.columns
x= db['YearsExperience'].values.reshape(30,1)
x
y=db['Salary']
y
from sklearn.linear_model import LinearRegression
mind =LinearRegression()
mind.fit(x,y)
LinearRegression()
import joblib
joblib.dump(mind,'trained.sk1')
mind.coef_
mind.predict([[1.3]])
mind.intercept_