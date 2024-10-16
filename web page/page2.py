import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import GaussianNB


     
st.set_page_config(page_title="GetHeartSure", page_icon="🫀", layout="centered", initial_sidebar_state="collapsed")




# Load the dataset
def load_data():
    df = pd.read_csv(f"/Users/harshareddy/Downloads/Heart_Disease_Prediction.csv")  # Replace "heart_disease_dataset.csv" with your dataset file
    return df

# Train the model
@st.cache_data
def train_model(df):
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr_model=LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_pred=lr_model.predict(X_test)
    lr_accur=accuracy_score(y_test,lr_pred)
    
    svc_model=SVC(probability=True)
    svc_model.fit(X_train,y_train)
    svc_pred=svc_model.predict(X_test)
    svc_accur=accuracy_score(y_test,svc_pred)
    
    knn_model=KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    knn_pred=knn_model.predict(X_test)
    knn_accur=accuracy_score(y_test, knn_pred)
    
    dt_model=DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred=dt_model.predict(X_test)
    dTaccur=accuracy_score(y_test, dt_pred)
    
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accur = accuracy_score(y_test, rf_pred)
    return  lr_model,lr_accur,svc_model ,svc_accur,knn_model,knn_accur,dt_model,dTaccur,rf_model,rf_accur


# Main function to run the app
def main():
    st.title('Heart Disease Prediction')

    # Load data
    df = load_data()

    # Train model
    lr_model,lr_accur,svc_model ,svc_accur,knn_model,knn_accur,dt_model,dTaccur,rf_model,rf_accur= train_model(df)

    # Sidebar with user input
    st.sidebar.header('User Input Features')

    age = st.sidebar.slider('Age', 20, 100, 50)
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type', ['1', '2', '3', '4'])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    chol = st.sidebar.slider('Cholesterol Level (mg/dl)', 100, 600, 200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', ['YES', 'NO'])
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', ['0', '1', '2'])
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 60, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ['YES', 'NO'])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', ['1', '2', '3'])
    ca = st.sidebar.slider('Number of Major Vessels Colored by Flourosopy', 0, 4, 0)
    thal = st.sidebar.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    # Preprocess user input
    sex = 0 if sex == 'Male' else 1
    fbs = 1 if fbs == 'Yes' else 0
    exang = 1 if exang == 'Yes' else 0
    thal = 0 if thal == 'Normal' else (1 if thal == 'Fixed Defect' else 2)

    # Create DataFrame for prediction
    user_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    # Display user input
    st.subheader('User Input:')
    st.write(user_data)
    
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    est_list=[
    ('knn',knn_model),
    ('lc',lr_model),
    ('dt',dt_model),
    ('rfc',rf_model),
    ('svc',svc_model)]
    s_m=StackingClassifier(
    estimators=est_list, final_estimator=GaussianNB())
    s_m.fit(X_train,y_train)
    y_t_p=s_m.predict(X_train)
    y_te_p=s_m.predict(X_test)
    s_m_t=accuracy_score(y_train,y_t_p)
    s_m_te=accuracy_score(y_test,y_te_p)
    
 # Make prediction
 
    #lr_prediction = lr_model.predict(user_data)
    #svc_prediction = svc_model.predict(user_data)
    #rf_prediction = rf_model.predict(user_data)
    #dt_prediction = dt_model.predict(user_data)
    #knn_prediction = knn_model.predict(user_data)
    #prediction=dt_model.predict(user_data)
    s_m_prediction= s_m.predict(user_data)
    
    # Assuming lr_model, svc_model, rf_model, dt_model, knn_model are your trained models
    #lr_prediction_proba = lr_model.predict_proba(user_data)
    #svc_prediction_proba = svc_model.predict_proba(user_data)
    #rf_prediction_proba = rf_model.predict_proba(user_data)
    #dt_prediction_proba = dt_model.predict_proba(user_data)
    #knn_prediction_proba = knn_model.predict_proba(user_data)
    s_m_prediction_proba = s_m.predict_proba(user_data)
    #prediction_proba=dt_model.predict_proba(user_data)

    print("Prediction probabilities:", s_m_prediction_proba)
    print("Predicted class label:", s_m_prediction[0])
   
    st.subheader('Prediction:')
    if s_m_prediction[0] == 0:
        st.success('Congratulations,You have No Heart Disease')
    else:
        st.warning(' Oops ! You might have Heart Disease')



    st.subheader('Prediction Probability:')
    st.write(f'Probability of No Heart Disease: {s_m_prediction_proba[0][0]:.2f}')
    st.write(f'Probability of Heart Disease: {s_m_prediction_proba[0][1]:.2f}')

    st.subheader(f'Model Accuracy: {s_m_t:.2f}')
   
   

  
      
if __name__ == '__main__':
    main()
