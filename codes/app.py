# from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
# import numpy as np
from joblib import load
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#strealit update: binary to text buffer
import io

st.set_option('deprecation.showfileUploaderEncoding', False)

# model = load_model('deployment_8082020')
model = load('ins_linreg.joblib') 

def predict(model, input_df):
    # predictions_df = predict_model(estimator=model, data=input_df)
    predictions = model.predict(input_df)
    # predictions = predictions_df['Label'][0]
    #get rid of unfeasible predictions
    predictions[predictions<0] = 0.00
    return predictions

def run():

    from PIL import Image
    #image = Image.open('logo.png')
    image_hospital = Image.open('hospital.jpg')

    # st.image(image,use_column_width=False)

    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?",
    ("Online", "Batch"))

    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('https://github.com/memoatwit/dsexample/')
    
    st.sidebar.image(image_hospital)

    st.title("Insurance Charges Prediction App")
####################
#Online
####################
    if add_selectbox == 'Online':

        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Sex', ['male', 'female'])
        bmi = st.number_input('BMI', min_value=10, max_value=50, value=10)
        children = st.selectbox('Children', [0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region', ['southwest', 'northwest', 'northeast', 'southeast'])

        output=""

        input_dict = {'age' : age, 'sex' : sex, 'bmi' : bmi, 'children' : children, 'smoker' : smoker, 'region' : region}
        input_df = pd.DataFrame([input_dict])
        if st.button("Predict"):
            output = predict(model=model, input_df=input_df)[0]
            output = 0 if output<0  else output 
            output = '$' + str(round(output,2))

        st.success('The output is {}'.format(output))
####################
#BATCH
####################

    if add_selectbox == 'Batch':
        try:
            file_buffer = st.file_uploader("Upload CSV file with 6 features: age, sex, bmi, children, smoker, region", type=["csv"])
            bytes_data = file_buffer.read()
            s=str(bytes_data)
            file_upload = io.StringIO(s)
        #strealit update: binary to text buffer
#         file_upload = io.TextIOWrapper(file_buffer)
#         if file_upload is not None:
#         try:
            data = pd.read_csv(file_upload)
            ## predictions = predict_model(estimator=model,data=data)
            # predictions = model.predict(data)
            # predictions[predictions<0] = 0.
            predictions = predict(model=model, input_df=data)
            st.write(predictions)
#         else:
#             st.write("else")
#             continue
        except:
            st.write("Please upload a valid CSV file.")
    # st.write("The pycaret post that inspired this project:")
    # st.write("https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104")
    st.markdown(
    """<a href="https://towardsdatascience.com/build-and-deploy-machine-learning-web-app-using-pycaret-and-streamlit-28883a569104">The pycaret post that inspired this project.</a>""", unsafe_allow_html=True,
)

            
if __name__ == '__main__':
    run()
