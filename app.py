import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline
from smote_variants import SMOBD
import pickle
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif




app = Flask(__name__)
model = pickle.load(open("XGBSMOTE_IPF_pretrained_pipeline.sav", 'rb')) # model[0] contains the model and model[1] contains support vector of feature matrix 


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]

    # final_features = [np.array(int_features)]
    final_features = pd.DataFrame(data = int_features, dtype = float).T
    final_features.columns = ['x', 'y', 'z', 'n_points', 'n_order', 'volume', 'positive_volume',
       'negative_volume', 'area', 'class_1_mean', 'class_1_sigma',
       'class_2_mean', 'class_2_sigma', 'class_3_mean', 'class_3_sigma',
       'class_4_mean', 'class_4_sigma', 'n_orientations_origin',
       'intensity_mean_origin', 'intensity_sigma_origin',
       'intensity_mean_destination', 'intensity_sigma_destination',
       'correlation_index', 'orientation_mean_origin', 'slope_mean_origin',
       'orientation_mean_destination', 'slope_mean_destination',
       'coplanararity_index_mean_origin', 'coplanararity_index_sigma_origin',
       'colinearity_index_mean_origin', 'colinearity_index_sigma_origin',
       'coplanararity_index_mean_destination',
       'coplanararity_index_sigma_destination',
       'colinearity_index_mean_destination',
       'colinearity_index_sigma_destination', 'angles_mean', 'angles_sigma']


    # uncomment the below if you have the dataset
    # mean_X = np.round(np.mean(final_features), 10)
    # mean_X = np.abs(mean_X)
    # std_X = np.round(np.std(final_features), 3)

    # print ("\nMean:\t", mean_X, "\nStd:\t", std_X)

    # if (np.round(mean_X.all(), 0) != 0) or (np.round(std_X.all(), 0) != 1):
    #     print ("Normalization!")
    #     scaler = StandardScaler()
    #     X = pd.read_csv("data/Degotalls_X_ALL.csv", index_col = 0)
    #     scaler.fit(X)
    #     final_features = scaler.transform(final_features)
    #     #final_features = (final_features - final_features.mean()) / final_features.std() # data normalization
    # else:
    #     print ("Data is normalized!")


    print('Dataframe: \t', final_features)
    final_features = np.array(final_features)
    xgb_features = final_features.T[model[1]] 
    xgb_features = xgb_features.T
    prediction = model[0].predict(xgb_features)
    output = prediction[0]

    return render_template('index.html', prediction_text=' Candidate for Rockfall: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
