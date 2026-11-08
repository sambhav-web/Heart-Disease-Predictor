import streamlit as st
import pandas as pd
import joblib
with open('one_hot_encoder.pkl','rb') as f:
  onehot=joblib.load(f)
with open('label_encoder.pkl','rb') as f:
  label=joblib.load(f)
with open('model.pkl','rb') as f:
  model=joblib.load(f)
st.title('Heart Disease Prediction')
age=st.slider('Age',min_value=29,max_value=77,step=1)
sex=st.selectbox('Sex',('M','F'))
cp=st.selectbox('ChestPainType',('ATA','NAP','ASY','TA'))
restbp=st.slider('RestingBP',min_value=94,max_value=200,step=1)
chol=st.slider('Cholesterol',min_value=126,max_value=564,step=1)
fbs=st.selectbox('FastingBS',('0','1'))
hr=st.slider('MaxHR',min_value=71,max_value=202,step=1)
oldpeak=st.slider('Oldpeak',min_value=0.0,max_value=6.2,step=0.1)
angina=st.selectbox('ExerciseAngina',('N','Y'))
restecg=st.selectbox('RestingECG',('Normal','ST','LVH'))
slope=st.selectbox('ST_Slope',('Up','Flat','Down'))
def check_disease(age,sex,cp,restbp,chol,fbs,hr,oldpeak,angina,restecg,slope):
  user=pd.DataFrame({
      'Age':[age],
      'Sex':[sex],
      'RestingBP':[restbp],
      'Cholesterol':[chol],
      'FastingBS':[fbs],
      'MaxHR':[hr],
      'Oldpeak':[oldpeak],
      'ExerciseAngina':[angina],
      'RestingECG':[restecg],
      'ST_Slope':[slope],
      'ChestPainType':[cp]
  })
  features=onehot.transform(user[['ChestPainType','RestingECG','ST_Slope','ExerciseAngina']])
  features=pd.DataFrame(features,columns=onehot.get_feature_names_out(),index=user.index)
  user=pd.concat([user.drop(columns=['ChestPainType','RestingECG','ST_Slope','ExerciseAngina']),features],axis=1)
  user['Sex']=label.transform(user['Sex'])
  pred=model.predict(user)[0]
  return pred
if st.button('Predict'):
  pred=check_disease(age,sex,cp,restbp,chol,fbs,hr,oldpeak,angina,restecg,slope)
  if pred==1:
    st.error('Heart Disease Detected')
  else:
    st.success('No Heart Disease Detected')