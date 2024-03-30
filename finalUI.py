import streamlit as st
import numpy as np
import pickle
import sklearn

with open('logistic_model_final.pkl', 'rb') as file:
    model = pickle.load(file)
    


def predict_property_damage(features):
    return model.predict([features])

def main():
    

    
    st.title("Heart Disease Risk Prediction")
    
    
    st.write("""
        This app can help predict your risk of heart disease. 
    """)
    st.write("""
        This program is based on 300,000 pieces of physical health data and trained on logistic regression models. After comparing multiple models such as Decision Tree, Random Forest, K-Nearest Neighbor, GaussianNB, and comparing multiple methods such as PCA and smote , the logistic regression model with the highest F1 score was selected for training and prediction.
    """)
    
    st.image('HeartDiseaseRisk.jpg', caption='Heart Disease Risk Prediction')

    st.title("Top factors associated with Heart Disease Risk Prediction")
    
    st.write("""
        According to the comparison between EDA and the 5 models, among the 18 features, "Age_Category", "Diabetes" and "Sex" are the two main causes of Heart_Disease.
    """)
    
    st.image('finalplot1.png', caption='Feature Importances for Logistic Regression')

    
    st.image('plot2.JPG', caption='Correlation heatmap')
    st.image('plot3.png', caption='ROC')
    
    
    st.title("Logistic regression model prediction, F1 Score: 0.8839 ")
    
    


    
    general_health = st.selectbox('General Health', options=[0, 1, 2, 3, 4], format_func=lambda x: ["Excellent", "Fair", "Good", "Poor", "Very Good"][x])
    checkup = st.selectbox('Checkup', options=[0, 1, 2, 3, 4], format_func=lambda x: ["5 or more years ago", "Never", "Within the past 2 years", "Within the past 5 years", "Within the past year"][x])
    exercise = st.selectbox('Exercise', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    
    skin_cancer = st.selectbox('Skin Cancer', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    other_cancer = st.selectbox('Other Cancer', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    depression = st.selectbox('Depression', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    diabetes = st.selectbox('Diabetes', options=[0, 1, 2, 3], format_func=lambda x: ["No", "No, pre-diabetes or borderline diabetes", "Yes", "Yes, but female told only during pregnancy"][x])
    arthritis = st.selectbox('Arthritis', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    sex = st.selectbox('Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    age_category = st.selectbox('Age Category', options=range(13), format_func=lambda x: ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80+"][x])
    
    #
    height_cm = st.number_input('Height (cm)', min_value=0.0, value=0.0, step=0.1)
    weight_kg = st.number_input('Weight (kg)', min_value=0.0, value=0.0, step=0.1)
    bmi = st.number_input('BMI', min_value=0.0, value=0.0, step=0.1)
    smoking_history = st.selectbox('Smoking History', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

    alcohol_consumption = st.number_input('Alcohol Consumption', min_value=0.0, value=0.0, step=0.1)
    fruit_consumption = st.number_input('Fruit Consumption', min_value=0.0, value=0.0, step=0.1)
    green_vegetables_consumption = st.number_input('Green Vegetables Consumption', min_value=0.0, value=0.0, step=0.1)
    friedpotato_consumption = st.number_input('FriedPotato Consumption', min_value=0.0, value=0.0, step=0.1)
        

    if st.button('Predict'):
        
        features = [general_health, checkup, exercise, skin_cancer, other_cancer,depression, diabetes, arthritis, sex, age_category,height_cm, weight_kg, bmi, smoking_history, alcohol_consumption, fruit_consumption,green_vegetables_consumption, friedpotato_consumption]
        
       
        prediction = predict_property_damage(features)
        
        
        result = "High" if prediction[0] == 1 else "Low"
        

        
        st.markdown(f"<h3 style='color: black;font-size: 24px'>Based from the Machine Learning model, your risk of developing Cardiovascular Disease (CVD) is:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: red;font-size: 30px'>Risk: {result}</h3>", unsafe_allow_html=True)

                # Provide safety recommendations
        if prediction[0] == 1:
            st.warning("High risk predicted. Please consider the following prevention measures:")
            st.write("- Avoid sedentary lifestyles")
            st.write("- Be careful when you have Smoking History")
            st.write("- Comorbidities: The presence of comorbidities like arthritis, diabetes, depression, skin cancer, and other cancers was associated with an increased risk of heart disease. These conditions may share underlying biological mechanisms or contribute to lifestyle factors that exacerbate cardiovascular risk.")
            st.write("- Remember regular Checkup")
        else:
            st.success("Lower risk predicted. Keep alarm and consider the following prevention measures :")
            st.write("- Avoid sedentary lifestyles")
            st.write("- Remember regular Checkup")

        
        
        
        
if __name__ == "__main__":
    main()
    