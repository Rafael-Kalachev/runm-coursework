# import pandas lib as pd
import pandas as pd
import os 

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import config as _conf
import int_util as _util

import pickle


def cleanup_dataset(df_orig):
    # type (DataFrame) -> DataFrame
    """
    Produce a new dataset, that is cleaned up and prepared for training.

    """
    _util.log("Data Prep and Cleanup", prepend="# ")

    _util.log(df_orig.head())



    _util.log("Empty cells count \n"
          "====================\n"
          "{}".format(df_orig.isnull().sum())) 
    
    df = df_orig.copy()

    _util.log("After investigation we see that for these columns the 'no' value is skipped, so we fill it in.\n")
    df = df.fillna("no")

    _util.log("Empty cells count (after filling the 'no' navlues) \n"
          "====================\n"
          "{}".format(df.isnull().sum())) 
    
    _util.log("No empty values.")


    _util.log("Datatypes of each column: \n"
              "===========================\n")
    _util.log(df.dtypes)

    _util.log("Lets look at the unique values of the int data")
    for col in df:
        if df[col].dtype  == "int64":
            unique_values = df[col].unique()

            _util.log("{}, {}".format(col,
                                          unique_values))
    
    colums_to_tf = ['GENDERN', 'INC_HUNGERN', 'WGHT_LOSSN', 'FATIGUEN',
                    'BLUR_VISIONN', 'SLOW_HEALINGN', 'FREQ_INFECTIONSN']
    for col in colums_to_tf:
        df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())


    for col in df:
        if df[col].dtype  == "object":
            df[col] = df[col].str.lower()
            unique_values = df[col].unique()
            to_be_dropped = False
            if unique_values.size <= 1:
                to_be_dropped =True
            _util.log("{}, {} {}".format(col,
                                          unique_values, 
                                          "  ((Drop))" if to_be_dropped else ""))
            if to_be_dropped:
                df = df.drop(columns=[col])


    _util.log("We notice that all the columns iwth name 'UNIT_*' contain values that are the same.\n"
              "i.e. the unit in which the previous value is measured.\n"
              "We also notice that all values are measured in the same units so we can drop these columns\n")


    #_util.log(" - We also notice  that the DIABETESN field is expressed in classes in the DIABETES column, thus it can be dropped too.\n")
    df = df.drop(columns=["DIABETESN"])
    _util.log(" - We also notice  that the SUBJID is present, which is just the ID of the patient, thus is irrelevant for the outcome, thu dropped too.\n")
    df = df.drop(columns=["SUBJID"])
    _util.log(" - We also notice  that the GENDER field is expressed in classes in the GENDERN column, thus it can be dropped too.\n")
    df = df.drop(columns=["GENDER"])
    _util.log(" - We also notice  that the ETHNICN field is expressed in classes in the ETHNIC column, thus it can be dropped too.\n")
    df = df.drop(columns=["ETHNICN"])

    _util.log("convert the yes/no classes to 0/1 values", prepend="## ")
    columns=["INC_THIRST",
             "FREQ_URIN",
             "INC_HUNGER",
             "WGHT_LOSS",
             "FATIGUE",
             "BLUR_VISION",
             "SLOW_HEALING",
             "FREQ_INFECTIONS"]
    
    for col in columns:
        df[col] = df[col].map({"yes":1, "no":0})

    # Create an instance of the OneHotEncoder
    encoder = OneHotEncoder()


    # Fit and transform the multi-class feature for both training and testing data
    df['ETHNIC'] = df['ETHNIC'].str.replace(' ', '_')
    encoded_feature = pd.get_dummies(df['ETHNIC'])

    for col in encoded_feature:
        encoded_feature[col] = encoded_feature[col].astype(bool).replace({False: 0, True: 1})


    _util.log(encoded_feature)

    df = df.drop(['ETHNIC'], axis=1)
    df = df.join(encoded_feature)
    df = pd.DataFrame(df)

    # Fit and transform the multi-class feature for both training and testing data
    df['DIABETES'] = df['DIABETES'].str.replace(' ', '_')
    encoded_feature = pd.get_dummies(df['DIABETES'])

    for col in encoded_feature:
        encoded_feature[col] = encoded_feature[col].astype(bool).replace({False: 0, True: 1})


    _util.log(encoded_feature)

    df = df.drop(['DIABETES'], axis=1)
    df = df.join(encoded_feature)
    df = pd.DataFrame(df)
    

    _util.log(df.head())

    return df

def in_out_split(df):
    in_df = df.iloc[:, 0:37]
    out_df = df.iloc[:, 38:]

    _util.log("out_d")
    _util.log(out_df)
    
    return in_df, out_df




def data_observation(df):
    
    _util.log("Data Observation", prepend="# ")

    for col in df:
        figure_name = "{col}_hist.png".format(col=col)
        figure_path = os.path.join(_conf.out_dir, figure_name)
        _util.log("![{col}_histogram]({figure_path})".format(col=col, figure_path=figure_path))
        fig, ax = plt.subplots()
        df.hist(column=col, figsize=(5,5), ax=ax)
        fig.savefig(figure_path)

    fig = plt.figure(figsize=(50,20))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
    figure_name = "df_correlation.png".format(col=col)
    figure_path = os.path.join(_conf.out_dir, figure_name)
    _util.log("![df_correlation]({figure_path})".format(col=col, figure_path=figure_path))
    fig.savefig(figure_path)
    


def main():
    _util.prepare_output_dir(_conf.out_dir)

    df_orig = _util.load_dataframe_from_file(_conf.database_file)

    _util.log("Dataset size: {}\n".format(_util.get_dataset_size(df_orig)))
    _util.log("")

    #_util.log("D")
    pd.set_option('display.max_columns', None)

    df = cleanup_dataset(df_orig)

    data_observation(df)
    
    X, y = in_out_split(df)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=_conf.train_test_split_percent, 
                                                        random_state=42)
    
    
    
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create a multioutput regression model using Keras
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(3)  # 3 output neurons for the 3 target variables
        ])

    #   Compile the model with an appropriate loss function and optimizer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Train the model on the training data
    model.fit(X_train_scaled, y_train, epochs=15, batch_size=32)

    # Evaluate the model on the testing data
    score = model.evaluate(X_test_scaled, y_test)
    print("Mean Squared Error:", score)

    model_name="dl_model.pkl"
    model_file_path = os.path.join(_conf.out_dir, model_name)
    with open(model_file_path, 'wb') as file:
        pickle.dump(model, file)

    # Get the predicted probabilities for the test data
    y_pred_prob = model.predict(X_test_scaled)

    # Convert predicted probabilities to class labels
    y_pred = np.argmax(np.asarray(y_pred_prob), axis=1)

    y_test=  np.argmax(np.asarray(y_test), axis=1)

    # Calculate the accuracy of the classifier
    acc = accuracy_score(y_test, y_pred)
    _util.log('Accuracy: {}'.format(acc))

    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    _util.log(conf_mat)

    fig = plt.figure(figsize=(20,10))
    sns.heatmap(conf_mat, annot=True)
    figure_name = "dl_confusion_matrix.png"
    figure_path = os.path.join(_conf.out_dir, figure_name)
    _util.log("![dl_confusion_matrix]({figure_path})".format(figure_path=figure_path))
    fig.savefig(figure_path)

    """
    sample = X_test_scaled[0:1000]

    explainer = shap.DeepExplainer(model, sample)

  
    feature_names= list(X_test.columns)
    shap_values = explainer.shap_values(sample)
    feature_importance = np.mean(np.abs(shap_values), axis=0)

    # Visualize the Shapley values for the chosen sample
    fig = plt.figure(figsize=(20,10))
    shap.force_plot(explainer.expected_value, shap_values[0], feature_names=feature_names)
    figure_name = "_shap_force_plot_0.png"
    figure_path = os.path.join(_conf.out_dir, figure_name)
    _util.log("![shap_force_plot_0]({figure_path})".format(figure_path=figure_path))
    fig.savefig(figure_path)

       # Visualize the Shapley values for the chosen sample
    fig = plt.figure(figsize=(20,10))
    shap.force_plot(explainer.expected_value, shap_values[1], feature_names=feature_names)
    figure_name = "_shap_force_plot_1.png"
    figure_path = os.path.join(_conf.out_dir, figure_name)
    _util.log("![shap_force_plot_1]({figure_path})".format(figure_path=figure_path))
    fig.savefig(figure_path)

       # Visualize the Shapley values for the chosen sample
    fig = plt.figure(figsize=(20,10))
    shap.force_plot(explainer.expected_value, shap_values[2], feature_names=feature_names)
    figure_name = "_shap_force_plot_2.png"
    figure_path = os.path.join(_conf.out_dir, figure_name)
    _util.log("![shap_force_plot_2]({figure_path})".format(figure_path=figure_path))
    fig.savefig(figure_path)

    """







if __name__ == "__main__":
    
    main()
