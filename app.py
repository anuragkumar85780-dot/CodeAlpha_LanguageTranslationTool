import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI FAQ Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI FAQ Chatbot")
st.write("Ask a question and the chatbot will find the most relevant answer.")

# -------------------------------------------------
# FAQ DATASET
# -------------------------------------------------

faq_data = {
    "question": [
        "What courses are available?",
        "How can I apply for admission?",
        "How do I get admission?",
        "What are the admission requirements?",
        "What are your working hours?",
        "When are you open?",
        "What time do you open?",
        "What time do you close?",
        "How can I contact support?",
        "How can I contact you?",
        "What is the fee structure?",
        "How much are the fees?",
        "Is online learning available?",
        "Do you offer online classes?",
        "How do I register?",
        "How can I register?",
        "What documents are required for admission?",
        "What documents do I need?",
        "Where is the college located?",
        "How can I reset my password?",
        "How can I get my student ID?"
    ],

    "answer": [
        "We offer undergraduate and postgraduate courses in multiple fields.",
        "You can apply for admission through the official admission process.",
        "You can apply for admission through the official admission process.",
        "Admission requirements depend on the selected course and program.",
        "Our working hours are 9:00 AM to 6:00 PM, Monday to Friday.",
        "Our working hours are 9:00 AM to 6:00 PM, Monday to Friday.",
        "We open at 9:00 AM from Monday to Friday.",
        "We close at 6:00 PM from Monday to Friday.",
        "You can contact our support team through the official support email.",
        "You can contact our support team through the official support email.",
        "The fee structure depends on the selected course and program.",
        "The fees depend on the selected course and program.",
        "Yes, selected courses and learning materials are available online.",
        "Yes, selected courses and learning materials are available online.",
        "You can register by completing the registration form and submitting the required information.",
        "You can register by completing the registration form and submitting the required information.",
        "You generally need identification, academic certificates and other documents specified during admission.",
        "You generally need identification, academic certificates and other documents specified during admission.",
        "The college location and directions are available through the official administration office.",
        "You can reset your password using the password reset option on the login page.",
        "You can collect or request your student ID through the student administration office."
    ]
}

faq_df = pd.DataFrame(faq_data)

# -------------------------------------------------
# TEXT CLEANING
# -------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


faq_df["clean_question"] = faq_df["question"].apply(clean_text)

# -------------------------------------------------
# TF-IDF MODEL
# -------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

faq_vectors = vectorizer.fit_transform(
    faq_df["clean_question"]
)

# -------------------------------------------------
# KEYWORD GROUPS
# -------------------------------------------------

keyword_groups = {
    "working_hours": [
        "open",
        "opening",
        "close",
        "closing",
        "hours",
        "time",
    ],

    "admission": [
        "admission",
        "apply",
        "application",
        "enroll",
        "enrollment"
    ],

    "courses": [
        "course",
        "courses",
        "program",
        "programs",
        "degree"
    ],

    "fees": [
        "fee",
        "fees",
        "cost",
        "price",
        "tuition"
    ],

    "support": [
        "support",
        "contact",
        "help",
        "email"
    ],

    "registration": [
        "register",
        "registration",
        "signup",
        "sign",
        "account"
    ],

    "documents": [
        "document",
        "documents",
        "certificate",
        "certificates",
        "paperwork"
    ],

    "online_learning": [
        "online",
        "remote",
        "virtual",
        "classes",
        "learning"
    ],

    "location": [
        "location",
        "located",
        "address",
        "where"
    ],

    "password": [
        "password",
        "login",
        "reset"
    ],

    "student_id": [
        "student",
        "id",
        "identity"
    ]
}

# -------------------------------------------------
# FAQ CATEGORY DETECTION
# -------------------------------------------------

def detect_category(question):

    question = clean_text(question)
    words = set(question.split())

    category_scores = {}

    for category, keywords in keyword_groups.items():

        score = 0

        for keyword in keywords:

            if keyword in words:
                score += 1

        category_scores[category] = score

    best_category = max(
        category_scores,
        key=category_scores.get
    )

    if category_scores[best_category] == 0:
        return None

    return best_category


# -------------------------------------------------
# GET ANSWER
# -------------------------------------------------

def get_answer(user_question):

    cleaned_question = clean_text(user_question)

    # First try category/keyword matching
    category = detect_category(cleaned_question)

    if category == "working_hours":
        return (
            "Our working hours are 9:00 AM to 6:00 PM, Monday to Friday.",
            1.00
        )

    if category == "admission":
        return (
            "You can apply for admission through the official admission process.",
            1.00
        )

    if category == "courses":
        return (
            "We offer undergraduate and postgraduate courses in multiple fields.",
            1.00
        )

    if category == "fees":
        return (
            "The fee structure depends on the selected course and program.",
            1.00
        )

    if category == "support":
        return (
            "You can contact our support team through the official support email.",
            1.00
        )

    if category == "registration":
        return (
            "You can register by completing the registration form and submitting the required information.",
            1.00
        )

    if category == "documents":
        return (
            "You generally need identification, academic certificates and other documents specified during admission.",
            1.00
        )

    if category == "online_learning":
        return (
            "Yes, selected courses and learning materials are available online.",
            1.00
        )

    if category == "location":
        return (
            "The college location and directions are available through the official administration office.",
            1.00
        )

    if category == "password":
        return (
            "You can reset your password using the password reset option on the login page.",
            1.00
        )

    if category == "student_id":
        return (
            "You can collect or request your student ID through the student administration office.",
            1.00
        )

    # TF-IDF fallback
    user_vector = vectorizer.transform(
        [cleaned_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )[0]

    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[best_match_index]

    if best_score < 0.20:
        return (
            "Sorry, I could not find a relevant answer. "
            "Please try asking your question in a different way.",
            best_score
        )

    answer = faq_df.iloc[
        best_match_index
    ]["answer"]

    return answer, best_score


# -------------------------------------------------
# CHAT INTERFACE
# -------------------------------------------------

user_question = st.text_input(
    "Ask your question:",
    placeholder="Example: When are you open?"
)

if st.button(
    "🔍 Ask",
    use_container_width=True
):

    if not user_question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Finding the best answer..."
        ):

            answer, score = get_answer(
                user_question
            )

        st.subheader(
            "🤖 Chatbot Response"
        )

        st.success(answer)

        st.caption(
            f"Similarity Score: {score:.2%}"
        )


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.caption(
    "CodeAlpha Artificial Intelligence Internship Project — Task 2"
)