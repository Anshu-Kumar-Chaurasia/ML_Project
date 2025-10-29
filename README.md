# 😃 Emoji Prediction Using Machine Learning

### 📘 Overview

This project aims to predict the most appropriate emoji for a given text or sentence using Natural Language Processing (NLP) and Machine Learning techniques.
By analyzing the semantic meaning of text with GloVe word embeddings, the model learns emotional and contextual patterns, enabling it to suggest the emoji that best matches the sentence’s tone and meaning.

### 🧠 Key Features

Predicts emojis based on textual input

Uses GloVe pre-trained word embeddings for feature representation

Implements a deep learning model (LSTM / simple neural network) for accurate classification

Achieves meaningful text-to-emoji mapping using semantic similarity

Includes both Jupyter Notebook (.ipynb) and Python script (.py) versions

### 📂 Dataset

The project uses two CSV files:

train_emoji.csv — Training dataset (text and corresponding emoji labels)

test_emoji.csv — Testing dataset for evaluating model performance

#### Each record consists of:

Column	Description
Text	The input sentence
Emoji	The emoji representing the emotional meaning of the text
⚙️ Files Included

Emoji Prediction Using Machine Learning.ipynb — Main Jupyter Notebook containing the model and workflow

Emoji Prediction Using Machine Learning.py — Python script version of the notebook

glove.6B.50d.txt — GloVe pre-trained embeddings file for NLP representation

train_emoji.csv — Training dataset

test_emoji.csv — Testing dataset

### 🧩 Technologies Used

Python 3

NumPy, Pandas — Data processing

Matplotlib, Seaborn — Visualization

TensorFlow / Keras — Model building and training

GloVe — Word embeddings for semantic representation

### 🚀 How It Works

Data Preprocessing — Cleans text and handles missing values.

Word Embedding — Converts text into numerical vectors using GloVe embeddings.

Model Training — Trains an LSTM/Neural Network on the processed data.

Evaluation — Tests the model using unseen data and measures accuracy.

Prediction — Input any custom text to predict the most suitable emoji!

### 📈 Results

The model achieves high accuracy in mapping emotions and context to emojis, demonstrating strong understanding of text semantics and sentiment.

### 🧪 Example
Input Sentence	Predicted Emoji
“I love this movie!”	❤️
“I’m feeling sleepy.”	😴
“That’s so funny!”	😂

![WhatsApp Image 2025-10-28 at 22 30 55_c757d5c1](https://github.com/user-attachments/assets/eda2778d-d11e-4e8f-8250-26e7fed84bd5)
![WhatsApp Image 2025-10-28 at 22 30 56_00e59026](https://github.com/user-attachments/assets/81cc314c-4da7-4964-b45a-5480b1c76dd5)
![WhatsApp Image 2025-10-28 at 22 30 56_890504aa](https://github.com/user-attachments/assets/e1354bc4-a6a8-4a31-92b2-e68dabdbcd99)
![WhatsApp Image 2025-10-28 at 22 30 57_ec6a969a](https://github.com/user-attachments/assets/2d612816-be8f-42e6-a6a6-07b20cdaef11)


### 💻 How to Run
🔧 Prerequisites

Make sure you have Python 3.8+ installed.

#### 📦 Step 1: Clone the Repository
git clone https://github.com/yourusername/emoji-prediction-ml.git
cd emoji-prediction-ml

#### 🧰 Step 2: Install Dependencies

If a requirements.txt file is available:

pip install -r requirements.txt


Or manually install key libraries:

pip install numpy pandas matplotlib seaborn tensorflow keras

#### 🧠 Step 3: Run the Notebook

Open Jupyter Notebook and execute:

jupyter notebook "Emoji Prediction Using Machine Learning.ipynb"


Or run the Python script directly:

python "Emoji Prediction Using Machine Learning.py"

#### 🧾 Step 4: Test Your Own Sentences

Once the model is trained, you can input any custom text and get the predicted emoji instantly.

### 🏁 Conclusion

This project showcases how NLP and deep learning can be combined to interpret human language and emotions, bridging communication between text and expression through emoji prediction.
💡 Future Enhancements

Add more emojis and expand dataset diversity

Fine-tune embeddings with contextual models like BERT

Deploy as a web app or chatbot for real-time emoji prediction
