# 🎵 Track Score Prediction - Most Streamed Spotify Songs 2024

This project predicts the ***Track Score*** (a proxy for song popularity) based on multi-platform performance data using machine learning and deep learning models. The dataset includes features from platforms like **Spotify, TikTok, YouTube, Apple Music, Deezer, Amazon Music, Shazam, and SiriusXM**.

---

## 📊 Dataset

**Source**: [Most Streamed Spotify Songs 2024 - Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/most-streamed-spotify-songs-2024)

The dataset contains streaming and engagement statistics for the most popular songs in 2024.  
**Target Variable**: `Track Score` (manually derived to reflect popularity)

### ✅ Features Used

- `Spotify Streams`
- `Spotify Playlist Count`
- `Spotify Playlist Reach`
- `Spotify Popularity`
- `YouTube Likes`
- `TikTok Posts`
- `TikTok Likes`
- `TikTok Views`
- `YouTube Playlist Reach`
- `Apple Music Playlist Count`
- `AirPlay Spins`
- `SiriusXM Spins`
- `Deezer Playlist Count`
- `Deezer Playlist Reach`
- `Amazon Playlist Count`
- `Shazam Counts`

---

## 🧠 Models Trained

| Model                    | R² Score | MSE       |
|-------------------------|----------|-----------|
| Gradient Boosting       | **0.7831** | **973.94** |
| XGBoost Regressor       | 0.7646   | 1057.19   |
| TensorFlow Neural Net   | 0.7178   | 1267.33   |
| Random Forest Regressor | 0.6897   | 1393.44   |

### ✅ Best Model: Gradient Boosting Regressor

This model achieved the best performance overall, capturing the complex nonlinear patterns between a song’s multi-platform presence and its popularity score.

---

## 📈 Visualizations

Predicted vs Actual Track Scores:
- Scatter plots for each model
- Ideal 45° reference line shows model accuracy
- Streamlit app supports live visualization (`st.pyplot`)

---

## 🚀 Technologies Used

- Python 3.12
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib & Seaborn
- Pandas / NumPy
- Streamlit (for app interface)

---

## 🧪 How to Run

1. Clone the repo:
    ```bash
    git clone https://github.com/0quaaD/Spotify-2024-API.git
    cd Spotify-2024-API
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app (if using Streamlit):
    ```bash
    streamlit run app.py
    ```

---

## 📌 Conclusion

This project demonstrates how digital engagement across multiple platforms can be used to predict the success of music tracks. The results show that metrics from platforms like Spotify, TikTok, and YouTube strongly correlate with track scores. These models can support industry applications like trend forecasting, playlist optimization, and artist marketing strategies.

---

## 📬 Contact

For questions or collaboration:
- GitHub: [0quaaD](https://github.com/0quaaD)
- Email: eln6436@gmail.com

