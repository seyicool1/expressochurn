import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv('expresso_processed.csv')
df = data.copy()

st.markdown("<h1 style = 'text-align: center; color: #176B87'>EXPRESSOR PREDICTOR </h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'text-align: center; top-margin: 0rem; color: #64CCC5'>BUILT BY SEYI OLORUNHUNDO</h1>", unsafe_allow_html = True)



st.image('pngwing.com (6).png', width = 350, use_column_width= True )


st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style='color: #1F4172; text-align: center; font-family: Arial, sans-serif;'>Project Overview</h4>", unsafe_allow_html=True)
st.markdown("<p>The telecom predictive modeling project focuses on leveraging advanced machine learning to develop a robust model for accurately predicting customer attrition. Through in-depth analyses of historical data, identification of key influencing factors, and the use of advanced classification algorithms, the project aims to provide valuable insights for business analysts, entrepreneurs, and businesses of various scales. Its core objective is to create a versatile model that adapts seamlessly to diverse business scenarios, offering meaningful predictions based on crucial factors like location, income ($), client tenure, and others. This effort seeks to empower businesses with a potent tool for anticipating customer behaviors, contributing to strategic decision-making, and fostering sustainable growth and success. </p>", unsafe_allow_html= True)
st.sidebar.image('pngwing.com (5).png' ,width = 150, use_column_width = True, caption= 'Welcome User')
st.markdown("<br>", unsafe_allow_html = True)

encoder = LabelEncoder()
scaler = StandardScaler()

df.drop(['Unnamed: 0', 'MRG'], axis = 1, inplace = True)

for i in df.drop('CHURN', axis = 1).columns:
    if df[i].dtypes == 'O':
        df[i] = encoder.fit_transform(df[i])
    else:
        df[i] = scaler.fit_transform(df[[i]])

x = df.drop('CHURN', axis = 1)
y = df.CHURN

xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size= 0.20, stratify= y)

model = LogisticRegression()
model.fit(xtrain, ytrain)

st.title('EXPRESSO CHURN')
st.dataframe(data)


tenure = st.sidebar.selectbox('DURATION AS A CUSTOMER', data.TENURE.unique())
montant = st.sidebar.number_input('AMOUNT RELOADED', df.MONTANT.min(), df.MONTANT.max())
freq_rech = st.sidebar.number_input('RELOADS', df.FREQUENCE_RECH.min(), df.FREQUENCE_RECH.max())
revenue = st.sidebar.number_input('MONTHLY INCOME', df.REVENUE.min(), df.REVENUE.max())
arpu_segment = st.sidebar.number_input('INCOME(90 DAYS)', df.ARPU_SEGMENT.min(), df.ARPU_SEGMENT.max())
frequence = st.sidebar.number_input('INCOME FREQUENCY', df.FREQUENCE.min(), df.FREQUENCE.max())
data_volume = st.sidebar.number_input('ACTIVENESS OF CLIENT(90 DAYS)', df.DATA_VOLUME.min(), df.DATA_VOLUME.max())
no_net = st.sidebar.number_input('CALL DURATION', df.ON_NET.min(), df.ON_NET.max())
regularity = st.sidebar.number_input('REGULARITY', df.REGULARITY.min(), df.REGULARITY.max())




new_tenure = encoder.transform([tenure])

input_var = pd.DataFrame({'TENURE': [new_tenure],
                           'MONTANT': [montant], 
                           'FREQUENCE_RECH': [freq_rech],
                          'REVENUE':[revenue],
                           'ARPU_SEGMENT':[arpu_segment],
                            'FREQUENCE':[frequence],
                             'DATA_VOLUME':[data_volume],
                              'ON_NET':[no_net],
                                'REGULARITY':[regularity]})
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style= 'margin: -30px; color:olive; font:sans serif' >", unsafe_allow_html= True)
st.dataframe(input_var)

predicted = model.predict(input_var)
output = None
if predicted[0] == 0:
   output = 'Not Churn'
else:
    output = 'Churn'
# transformed= encoder.transform([predicted])
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The customer is predicted to {output}')
        st.balloons()




# age = st.slider('Age', data['age'].min(), data['age'].max())
# ba = st.select_slider('BA', data['ba'].unique())

# print(data.isnull().sum())