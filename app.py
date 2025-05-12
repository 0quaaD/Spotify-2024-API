import pandas as pd 
import pickle
import os
import warnings
import streamlit as st 
warnings.filterwarnings('ignore',category=FutureWarning)
df_cleaned = pd.read_csv('Dataset/Cleaned Data.csv',encoding='latin1',on_bad_lines='warn')
df_original = pd.read_csv("Dataset/Most Streamed Spotify Songs 2024.csv",encoding='latin1',on_bad_lines='warn')

st.header("🎧 Spotify 2024 API Dataset Machine Learning Models Comparison")
st.write("In this webpage, you will see the some machine learning models' comparison such as **Gradient Boosting Regressor, XGB Regressor, TensorFlow Keras and Random Forest Regressor** models.")
st.markdown("This is the dataset contents:")
st.markdown("""

    This project uses the **"Most Streamed Spotify Songs 2024"** dataset to build a regression model that predicts a song’s track score based on its audio characteristics and metadata. The dataset contains thousands of tracks and includes both popularity metrics and acoustic features provided by the Spotify API.
    
    🔍 Objective

    To predict the track score of a song using features such as:
       - Danceability
       - Energy
       - Loudness
       - Speechiness
       - Acousticness
       - Instrumentalness
       - Liveness
       - Valence
       - Tempo
       - Duration
       - Popularity
       - Genre

       Original Dataset it located at this link: [Most Streamed Spotify Songs 2024](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024)

       First 10 rows of the entire dataset:
    """)
st.dataframe(data=df_original.head(10))
st.header("📈 Training and Testing Models with 4 different Algorithms\n\n")
st.write("Now I would like to show you the models and their performances.")

with st.expander("**Gradient Boosting Model**"):
    st.markdown("What is Gradient Boosting Regression algorithm?\nIt may look similar to the Gradient Descent algorithm but it really doesn't. Gradient Boosting is used for series of weak learners such as Decision Trees. Also it's an ensemble learning method. However Gradient Descent is used for reducing loss value at Neural Networks for optimizations.")
    st.markdown("""
    $R^2$ Score of the Gradient Boosting Model : ~0.7831\n\n
    MSE of the model: ~973.94""")
    st.image('Model/grad_boost_importance.png')

    st.markdown("Now let's take a look at the prediction plot:")
    st.image('Model/grad_boost_pred.png')
    st.markdown("""As you can see, the model is pretty good for this messy data and made very good prediction.
    And the most important features for the model are **Spotify Popularity, Spotify Playlist Reach, Deezer Playlist Reach, Apple Music Playlist Count**, and so on so forth.
    Now let's see the results of the other models.""")

with st.expander("**XGB Regressor Model**"):
    st.markdown("""XGBoost is a powerful approach for building supervised regression models. The validity of this statement can be inferred by knowing about its (XGBoost) objective function and base learners. The objective function contains loss function and a regularization term. It tells about the difference between actual values and predicted values, i.e how far the model results are from the real values. The most common loss functions in XGBoost for regression problems is reg:linear, and that for binary classification is reg:logistics. Ensemble learning involves training and combining individual models (known as base learners) to get a single prediction, and XGBoost is one of the ensemble learning methods. XGBoost expects to have the base learners which are uniformly bad at the remainder so that when all the predictions are combined, bad predictions cancels out and better one sums up to form final good predictions.""")

    st.markdown("""$R^2$ Score of the XGB Regressor Model: ~0.7646\n\nMSE of the model: ~1057.19""")

    st.image('Model/xgb_importance.png')
    st.markdown("Now let's take a look at the prediction plot of the model:")
    st.image("Model/xgb_pred.png")
    st.markdown("""As you can see, it predicted very well but not as much as the Gradient Boosting Model. The most features for the model are **Amazon Playlist Count, Deezer Playlist Reach, Spotify Playlist Reach, Spotify Popularity**, and so on so forth.""")

with st.expander("**Random Forest Model**"):
    st.markdown("""
Random forest algorithms have three main hyperparameters, which need to be set before training. These include node size, the number of trees, and the number of features sampled. From there, the random forest classifier can be used to solve for regression or classification problems.

The random forest algorithm is made up of a collection of decision trees, and each tree in the ensemble is comprised of a data sample drawn from a training set with replacement, called the bootstrap sample. Of that training sample, one-third of it is set aside as test data, known as the out-of-bag (oob) sample, which we’ll come back to later. Another instance of randomness is then injected through feature bagging, adding more diversity to the dataset and reducing the correlation among decision trees. Depending on the type of problem, the determination of the prediction will vary. For a regression task, the individual decision trees will be averaged, and for a classification task, a majority vote—i.e. the most frequent categorical variable—will yield the predicted class. Finally, the oob sample is then used for cross-validation, finalizing that prediction.
""")

    st.markdown("""$R^2$ Score of the Random Forest Model: ~0.6897\n\nMSE of the model: ~1393.44""")
    st.image("Model/rand_forest_importance.png")
    st.markdown("""That is the results of the Random Forest Algorithm Model and that wasn't good enough to compete against two models we talked about. The most features for the model is **Amazon Playlist Count, Spotify Playlist Reach, Deezer Playlist Reach,Spotify Popularity**, and so on so forth.""")

with st.expander("**TensorFlow Keras Neural Network Model**"):
    st.markdown("""
A neural network is a machine learning program, or model, that makes decisions in a manner similar to the human brain, by using processes that mimic the way biological neurons work together to identify phenomena, weigh options and arrive at conclusions.

Every neural network consists of layers of nodes or artificial neurons, an input layer, one or more hidden layers, and an output layer. Each node connects to others, and has its own associated weight and threshold. If the output of any individual node is above the specified threshold value, that node is activated, sending data to the next layer of the network. Otherwise, no data is passed along to the next layer of the network.""")
    
    st.markdown("""$R^2$ Score of the Neural Network: ~0.7178\n\n(However the NN model actually updates the $R^2$ score due to lack of the data for NN's. But it changes the range of ~0.71-0.79.)\n\nMSE of the model: ~1267.33\n\n(This score also changes due to the dataset that isn't suitable for NN's and it changes the range of ~1250-1800 which isn't good enough for the Machine Learning Model.)""")
    st.image("Model/tensor_keras_importance.png")
    st.markdown("""As you can see this is the feature importance plot of the NN model and the most important features are **Spotify Streams, Spotify PLaylist Count, Spotify Popularity, TikTok Posts**, and so others. However some of the features aren't important to the model such as **Apple Music Playlist Count and TikTok Views**.""")

    st.image("Model/tensor_keras_pred.png")
    st.markdown("From the plot of the prediction of the model, we can easily see that the NN model didn't predict like the others due to the reasons I have said earlier already.")

st.header("🧾 Conclusion")
st.markdown("""In this project, we explored and modeled the track score prediction of Spotify’s most-streamed songs in 2024 using a range of machine learning techniques. We preprocessed the dataset, selected meaningful audio features, and built predictive models including:\n\n
- Gradient Boosting Regression
- Random Forest Regressor
- XGBoost Regressor
- TensorFlow Keras Neural Network\n\n
Among these models, the Gradient Boosting Regressor achieved the best performance with an R² score of 0.7831 and the lowest MSE of 973.95, indicating a strong ability to capture relationships between the song characteristics and their streaming popularity score. The TensorFlow Neural Network also performed competitively with an R² score of 0.7251, showing that deep learning methods can be a valuable alternative for capturing non-linear patterns in audio data.\n\n
The features that have been used, is spanned to multiple platforms,including:\n\n
- Spotify : Streams, Playlist Count, Playlist Reach Popularity
- Youtube : Likes, Playlist Reach
- TikTok : Posts, Likes, Views
- Apple Music, Deezer, Amazon Music: Playlist Counts and Reach
- Shazam and SiriusXM: Shazam Counts, AirPlay Spins, SiriusXM Spins\n\n
🔎 Model Performance:
| Model | $R^2$ Score | MSE |
|-------|-------------|------|
| Gradient Boosting | 0.7831 | 973.94 |
| XGBoost Regressor | 0.7646 | 1057.19 |
| Random Forest Regressor | 0.6897 | 1393.44 |
| TensorFlow Neural Network | 0.7178 | 1267.33|\n\n
The Gradient Boosting Regressor emerged as the top performer in terms of both R² and MSE, showing it captured the complex relationships between multi-platform performance indicators and the track score.

These findings suggest that cross-platform engagement metrics — especially from Spotify, TikTok, and YouTube — play a significant role in determining a song's success. Such models could be used in industry applications for trend forecasting, playlist curation, and marketing strategy.
""")

