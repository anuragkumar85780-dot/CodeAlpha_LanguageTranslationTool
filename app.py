import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="AI Language Translation Tool",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 AI Language Translation Tool")
st.write("Translate text between English and Hindi using AI.")

# Available languages
languages = {
    "English": "en",
    "Hindi": "hi"
}

# Translation models
MODELS = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en"
}


@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# Language selection
col1, col2 = st.columns(2)

with col1:
    source_language = st.selectbox(
        "Source Language",
        list(languages.keys())
    )

with col2:
    target_language = st.selectbox(
        "Target Language",
        list(languages.keys()),
        index=1
    )


# Text input
text = st.text_area(
    "Enter Text",
    height=180,
    placeholder="Type your text here..."
)


# Translation
if st.button("🔄 Translate", use_container_width=True):

    if not text.strip():
        st.warning("Please enter some text.")

    elif source_language == target_language:
        st.info("Source and target languages cannot be the same.")

    else:
        source_code = languages[source_language]
        target_code = languages[target_language]

        model_name = MODELS[(source_code, target_code)]

        try:
            with st.spinner("AI is translating..."):

                tokenizer, model = load_model(model_name)

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                outputs = model.generate(
                    **inputs,
                    max_length=512
                )

                translated_text = tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )

            st.subheader("### Translated Text")

            st.success(translated_text)

            st.text_area(
                "📋 Copy Translation",
                value=translated_text,
                height=120
            )

        except Exception as e:
            st.error(f"Translation failed: {e}")


st.markdown("---")

st.caption(
    "CodeAlpha Artificial Intelligence Internship Project"
)