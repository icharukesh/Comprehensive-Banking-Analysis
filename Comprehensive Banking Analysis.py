import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Loading the dataset
df=pd.read_csv("C:/Users/S.ICHARUKESH/Downloads/train.csv")
df.head()

#Exploratory Data Analysis(EDA)
df.info()

df.shape

df.isnull().sum()

df['Credit_Score'].value_counts()

#Label Encoding for Categorical Columns

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cate_col=df.select_dtypes(include='object').columns
for col in cate_col:
    df[col]=le.fit_transform(df[col])
    
#Standard Scaler for Numerical Columns

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
num_cols=df.select_dtypes(include=['int64','float64']).columns
df[num_cols]=scaler.fit_transform(df[num_cols])

df.head(3)

#Customer Segmentation(Clustering)

#Elbow method(K)
from sklearn.cluster import KMeans
clus_features=df[['Age','Annual_Income','Monthly_Inhand_Salary']]
k_rng=range(1,11)
sse=[]

for k in k_rng:
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(clus_features)
    sse.append(km.inertia_)
    
#Visialization for Elbow plot
plt.figure(figsize=(8,5))
plt.plot(k_rng,sse,marker='o')
plt.title("Elbow Method for K")
plt.xlabel("Number of clusters")
plt.ylabel("SSE")

#Customer Segmentation(Clustering)
k=3
km=KMeans(n_clusters=k,random_state=42)
df['Cluster']=km.fit_predict(clus_features)

#Visualizing Clustering

plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Annual_Income'],y=df['Monthly_Inhand_Salary'],hue=df['Cluster'],palette='tab10')
plt.title("Customer Segmentation : Annual_Income vs Monthly_Inhand_Salary")
plt.xlabel("Annual_Income")
plt.ylabel("Monthly_Inhand_Salary")
plt.show()

#Credit Risk Assessment : Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
class_features=df[['Credit_Utilization_Ratio','Num_of_Loan','Outstanding_Debt','Num_of_Delayed_Payment','Credit_Mix','Payment_Behaviour','Delay_from_due_date','Changed_Credit_Limit','Payment_of_Min_Amount','Monthly_Balance']]
x_class=class_features
y_class=df['Credit_Score']

X_train, X_test, y_train, y_test =train_test_split(x_class,y_class,test_size=0.2,random_state=42)

model_rf=RandomForestClassifier(n_estimators=50,random_state=42)
model_rf.fit(X_train,y_train)

model_rf.predict(X_test)

model_rf.score(X_test,y_test)

#Visualization for Classification

importance=model_rf.feature_importances_
feature_names=class_features.columns
plt.figure(figsize=(8,5))
sns.barplot(x=feature_names,y=importance,palette='coolwarm')
plt.title('Feature Importance in Credit Risk Assessment')
plt.xlabel('Feature_Names')
plt.ylabel('Importance')

#Performace Prediction:Regression

reg_features=df[['Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan','Credit_Utilization_Ratio','Outstanding_Debt','Total_EMI_per_month','Amount_invested_monthly']]
x_reg=reg_features
y_reg=df['Monthly_Balance']
X_train, X_test, y_train, y_test =train_test_split(x_reg,y_reg,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
model_le=LinearRegression()
model_le.fit(X_train,y_train)
y_pred= model_le.predict(X_test)

# Regression Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='blue')
plt.title('Actual vs Monthly Balance')
plt.xlabel('Actual Monthly Balance')
plt.ylabel('Predicted Monthly Balance')
plt.show()
