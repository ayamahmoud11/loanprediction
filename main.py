import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('loan_data - loan_data.csv').drop(columns=['Loan_ID'])

df['Dependents'].replace('3+', 4, inplace=True)

l_amount = df["LoanAmount"].mean()
df["LoanAmount"].fillna(l_amount, inplace=True)

df["Dependents"].fillna(0, inplace=True)

term = df["Loan_Amount_Term"].mean()
df["Loan_Amount_Term"].fillna(term, inplace=True)

hist = df["Credit_History"].mean()
df["Credit_History"].fillna(hist, inplace=True)

df["Gender"].fillna('no', inplace=True)

df["Married"].fillna('no', inplace=True)

df["Self_Employed"].fillna('no', inplace=True)


# convert categorical
encoded_columns = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
labelencoder_X = LabelEncoder()
for col in encoded_columns:
    df[col] = labelencoder_X.fit_transform(df[col])

# split into Independent and Dependent
X = df.drop(columns=['Loan_Status'])
Y = df["Loan_Status"]

if __name__ == '_main_':
    print(df.isna().sum())
print(df.head())

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

print(df.isna().sum())

# splitting data into training and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#Data scaling
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
scaler.fit(df)
scaled_features = scaler.transform(df)
fig = plt.figure(figsize =(10, 7))
plt.boxplot(scaled_features)
plt.show()