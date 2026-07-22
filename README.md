# Emotion Detection using AI

##  Project Overview
This project is part of my internship at **Skillerz Hub**.  
It focuses on building a machine learning model that classifies text into emotions such as **joy, sadness, anger, fear, love, and surprise**.  
The project uses **TF‑IDF features** with a **neural network** and includes a **bonus demo** powered by Hugging Face Transformers.

---

##  Features
- Preprocessing text data with TF‑IDF
- Neural network model built using TensorFlow/Keras
- Evaluation with classification report and confusion matrix
- Visualizations: dataset distribution, accuracy/loss curves, prediction histogram
- Bonus demo with Hugging Face pipeline for instant emotion detection

---

## Tech Stack
**Programming Language(s):** Python  
**Libraries/Tools:** TensorFlow, Keras, Scikit‑Learn, Matplotlib, Seaborn, Hugging Face Transformers  
**Environment:** Google Colab  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/your-repo/blob/main/Emotion_Detection_using_AI.ipynb)

---

## Visual Outputs

### Dataset Distribution
![Dataset Distribution](images/datasetdistribution.png)

### Confusion Matrix Heatmap
![Confusion Matrix](images/confusionmatrix.png)

### Model Accuracy
![Model Accuracy](images/modelaccuracy.png)

### Model Loss
![Model Loss](images/modelloss.png)

### Predicted Emotion Counts
![Predicted Emotion Counts](images/predictedcounts.png)

---

## Results
- **Accuracy:** ~87%  
- Balanced precision, recall, and F1‑scores across most emotion categories  
- Strong diagonal in confusion matrix showing correct predictions

---

## Bonus Demo
Quick emotion detection using Hugging Face pre‑trained model:

-python

from transformers import pipeline

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
print(emotion_model("I am really happy today!"))
