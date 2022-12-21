from distutils.command.upload import upload
from tkinter import Button
from PIL import Image
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns


st.markdown(
"""
# Nguyễn Văn Toàn
# Mssv: 20522028
## Linear Regression với Streamlit 
""")

uploaded_file = st.file_uploader("Dataset")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    with open('./'+uploaded_file.name, "wb") as f: 
        f.write(bytes_data)
    file = pd.read_csv(uploaded_file)
    # st.write(file.head())
    if st.checkbox('View dataset in table data format'):
        st.dataframe(file)
    header = file.columns.values
    X = pd.DataFrame()
    input_features = []
    dem = 0
    st.header("Input Features")
    cols = st.columns(4)
    for i in range(len(header)):
        cbox = cols[int(i/len(header)*4)].checkbox(header[i])
        if cbox:
            input_features.append(header[i])
            X.insert(dem,dem,file[header[i]])
            dem = dem + 1
    st.dataframe(X)

    options = [header for header in header if header not in input_features and file.dtypes[header] != 'object']
    st.header("Type of Splitting Data")
    split_type = st.selectbox(" ", ("Train-Test Split", "K-Fold Cross Validation"))
    # , label_visibility="collapsed"
    cols = st.columns(2)
    # with cols[0]:
        # st.header("Output Feature")
    output_feature = st.radio("Output Feature",options)
    st.write(file[output_feature].values)
    y = file[output_feature]
    # st.write(X.shape,' ',y.shape)
    encs = []
    Y = file[output_feature].to_numpy()
    XX = np.array([])
    enc_idx = -1  
    for feature in input_features:
        x = file[feature].to_numpy().reshape(-1, 1)
        if (file.dtypes[feature] == 'object'):
            encs.append(OneHotEncoder(handle_unknown='ignore'))
            enc_idx += 1
            x = encs[enc_idx].fit_transform(x).toarray()
        if len(XX)==0:
            XX = x
        else:
            XX = np.concatenate((XX, x), axis=1)
    if split_type == "Train-Test Split":
        st.subheader("Train/Test Split")
        train = st.slider("Train/Test Split",0,10,1)
        k = train / 10
        Run = st.button("Run")
        if Run:
            X_train, X_test, Y_train, Y_test = train_test_split(XX, Y, train_size=k, random_state=0)
            model = LinearRegression()
            model_fit = model.fit(X_train, Y_train)
            Y_pred = model_fit.predict(X_test)
            mae = mean_absolute_error(y_true=Y_test, y_pred=Y_pred)
            mse = mean_squared_error(y_true=Y_test, y_pred=Y_pred)
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()
            ax1.bar(np.arange(1) - 0.17, [mae], 0.2, label='MAE', color='yellow')
            plt.xticks(np.arange(1), [str(k)])
            plt.xlabel("Folds", color='yellow')
            plt.ylabel("Mean Absolute Error", color='maroon')
            ax2 = ax1.twinx()
            ax2.bar(np.arange(1) + 0.17, [mse], 0.2, label='MSE', color='green')
            plt.ylabel('Mean Squared Error', color='green')
            plt.title("EVALUATION METRIC")
            plt.savefig('chart.png')
            with open('model.pkl','wb') as f:
                pickle.dump(model_fit, f)
    else:
        st.header("Numbers of Fold")
        k_fold = st.selectbox(" ", range(2, X.shape[0]))
        # , label_visibility="collapsed"
        Run = st.button("Run")
        if Run: 
            kf = KFold(n_splits=k_fold, random_state=None)
            folds = [str(fold) for fold in range(1, k_fold+1)]
            mae = []
            mse = []
            for train_index, test_index in kf.split(XX):
                X_train, X_test = XX[train_index, :], XX[test_index, :]
                Y_train, Y_test = Y[train_index], Y[test_index]
                model = LinearRegression().fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                mae.append(round(mean_absolute_error(y_true=Y_test, y_pred=Y_pred), 2))
                mse.append(round(mean_squared_error(y_true=Y_test, y_pred=Y_pred), 2))
                with open('model.pkl','wb') as f:
                    pickle.dump(model, f)
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()
            ax1.bar(np.arange(len(folds)) - 0.17, mae, 0.2, label='MAE', color='yellow')
            plt.xticks(np.arange(len(folds)), folds)
            plt.xlabel("Folds", color='yellow')
            plt.ylabel("Mean Absolute Error", color='maroon')
            ax2 = ax1.twinx()
            ax2.bar(np.arange(len(folds)) + 0.17, mse, 0.2, label='MSE', color='green')
            plt.ylabel('Mean Squared Error', color='green')
            plt.title("EVALUATION METRIC")
            plt.savefig('chart.png')

    img = cv2.imread('chart.png')
    if img is not None: 
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))     

    # SET INPUT
    st.header("Test Model")    
    cols = st.columns(3)
    input = np.array([])
    enc_idx = -1
    for i in range(len(input_features)):
        if (file.dtypes[input_features[i]] == 'object'):
            x = cols[int(i/len(input_features)*3)].selectbox(input_features[i], file[input_features[i]].unique())
            enc_idx += 1 
            x = encs[enc_idx].transform([[x]]).toarray()
        else: 
            x = cols[int(i/len(input_features)*3)].text_input(input_features[i], 0)
            x = np.array([[float(x)]])
        if len(input) == 0: 
            input = x 
        else:
            input = np.concatenate((input, x), axis=1)

    # TEST MODEL
    btn_predict = st.button("Predict")
    pred_val = [0]
    if btn_predict: 
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            pred_val = model.predict(input)
    cols = st.columns(2)
    st.subheader("Predict Value")
    st.text_input(" ", value=round(pred_val[0],2), key=2)
    # , label_visibility='collapsed'