import re
import random
import spacy
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

st.set_page_config(
    page_title="Chatbot Analytics",
    page_icon="📊"
)

random_quotes = [
    "Change is inevitable. Instead of resisting it, you're better served simply going with the flow.",
    "Revenge is just the path you took to escape your suffering.",
    "Arrogance destroys the footholds of victory.",
    "Even if no one believes in you, stick out your chest and scream your defiance!",
    "Do not live bowing down. You must die standing up.",
    "A bond is like a pointillist painting. In order to see it in its entirety, you have to take a step back.",
    "If a miracle only happens once, then what is it called the second time?",
    "Cast off your fear. Look forward. Go forward. Never stand still. Retreat and you will age. Hesitate and you will die."]

st.sidebar.success(random.choice(random_quotes))

def tokenize_words(messages):
    document = nlp(text=messages)

    # Remove stop words
    filtered_tokens = [token.text for token in document if not token.is_stop]
    filtered_text = ' '.join(filtered_tokens)

    # Remove special characters
    cleaned_text = re.sub(r'[^a-zA-z0-9\s(\s)\[\]\{\}]', '', filtered_text)

    return cleaned_text.split()

nlp = spacy.load("en_core_web_sm")

"# Chatbot Analytics"

"""This page will display the tokenized words during your time with the Chatbot using graphs, such as Bar Chart and Word Cloud. You might discover new information with both your and the bot's responses from these visualizations."""

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2 = st.columns(2)

with col1:
    "## User Prompts"
    user_prompts = [message["content"] for message in st.session_state.messages if message["role"] == "user"]

    # Display counts as Bar Graph
    "### Bar Graph"
    if len(user_prompts) == 0:
        st.bar_chart({'words': [], 'counts': []}, x="words")
    else:
        document = tokenize_words(" ".join(user_prompts))
        keys, counts = np.unique(document, return_counts=True)

        st.bar_chart({'words': keys, 'counts': counts}, x="words")

    # Display counts as Word Cloud
    "### Word Cloud"
    wordcloud = WordCloud().generate_from_text("Missing")

    if len(user_prompts) > 0:
        wordcloud = WordCloud().generate_from_text(" ".join(document))

    wc_fig, wc_ax = plt.subplots(figsize=(12, 8))
    wc_ax.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(wc_fig)

with col2:
    "## Chatbot Responses"
    chatbot_responses = [message["content"] for message in st.session_state.messages if message["role"] == "assistant"]

    # Display counts as Bar Graph
    "### Bar Graph"
    if len(chatbot_responses) == 0:
        st.bar_chart({'words': [], 'counts': []}, x="words")
    else:
        document = tokenize_words(" ".join(chatbot_responses))
        keys, counts = np.unique(document, return_counts=True)

        st.bar_chart({'words': keys, 'counts': counts}, x="words")