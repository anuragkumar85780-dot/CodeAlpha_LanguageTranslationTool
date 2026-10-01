# 🤖 AI FAQ Chatbot

An AI-powered FAQ chatbot developed as part of the CodeAlpha Artificial Intelligence Internship — Task 2.

The chatbot uses Natural Language Processing (NLP) techniques to understand user questions and find the most relevant answer from a predefined FAQ dataset.

## 🚀 Features

- Interactive Streamlit chatbot interface
- FAQ-based question answering
- Natural Language Processing
- TF-IDF vectorization
- Cosine similarity
- Keyword-based question classification
- Similarity score display
- Handles unknown questions
- No external API or API key required

## 🧠 Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- TF-IDF
- Cosine Similarity

## ⚙️ How It Works

1. A predefined FAQ dataset contains questions and answers.
2. User enters a question through the Streamlit interface.
3. The question is cleaned and processed.
4. Keyword-based matching checks for common FAQ categories.
5. TF-IDF and cosine similarity provide a fallback matching method.
6. The chatbot returns the most relevant answer.

## 📂 Project Structure

```text
CodeAlpha_LanguageTranslationTool/
│
├── app.py
├── requirements.txt
├── README.md
└── venv/

