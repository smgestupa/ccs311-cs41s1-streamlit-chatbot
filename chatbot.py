import time
import random
import asyncio
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Ask about BLEACH カテゴリー",
    page_icon="💬"
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

async def every(__seconds: float, func, *args, **kwargs):
    while True:
        await asyncio.sleep(__seconds)
        func(*args, **kwargs)

def write_bot_message(response):
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ""

        for character in response:
            full_response += character
            time.sleep(0.025)

            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({'role': 'assistant', 'content': full_response})

def get_most_similar_response(df, query, top_k=1, index=0):
    if query == '':
        return

    # Prepare data
    vectorizer = TfidfVectorizer()
    all_data = list(df['Topics']) + [query]

    # Vectorize with TF-IDF
    tfidf_matrix = vectorizer.fit_transform(all_data)

    # Compute Similarity
    document_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]
    similarity_scores = cosine_similarity(query_vector, document_vectors)

    # Pick the Top k response
    sorted_indeces = similarity_scores.argsort()[0][::-1][:top_k]

    # Get the similarity score of the chosen response
    similarity_score = similarity_scores[0][similarity_scores.argsort()[0][::-1][:top_k]][0] * 100

    # Fetch the corresponding response
    most_similar_responses = df.iloc[sorted_indeces]['Responses'].values

    response = None if len(most_similar_responses) == 0 else most_similar_responses[index]
    response_index = df[df['Responses'] == response].index.item()

    return response, similarity_score

def suggest_topic(df, query):
    response, similarity_score = get_most_similar_response(df, query, top_k=20, index=random.randint(2, 19))

    if similarity_score >= 50.0:
        write_bot_message(f'Did you know? {response}')

def get_fallback_message():
    fallback_message = [
            'I don\'t understand your message, please try again.',
            'I need more information about what you want to know. Keep going!',
            'Hmmm... I may have limited information about your message, I am sorry.',
            'I don\'t know what you want to say. Apologies.'
        ]

    return random.choice(fallback_message)

topics_responses = 'https://raw.githubusercontent.com/smgestupa/ccs311-cs41s1-streamlit-chatbot/main/content/NLP-Chatbot-Data.csv'

chatdata_df = pd.read_csv(topics_responses)

async_loop = asyncio.new_event_loop()

"""# Ask about BLEACH カテゴリー"""

"""Bleach (stylized in all caps) is a Japanese anime television series based on Tite Kubo's original manga series of the same name. This chatbot is free for you to ask anything Bleach-related, whether you're a fan or not, I hope the chatbot will provide you with useful information."""

"""Please feel free to try this and hope you enjoy! 😊"""

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages.append({'role': 'assistant', 'content': get_most_similar_response(chatdata_df, "Help.")[0]})

last_query = ''

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Ask away!')

if prompt is not None:
    last_query = prompt

    with st.chat_message('user'):
        st.markdown(prompt)

    async_loop.create_task(every(15, suggest_topic, chatdata_df, last_query))

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    response, similarity_score = get_most_similar_response(chatdata_df, prompt)

    if similarity_score >= 50.0:
        write_bot_message(response)
    else:
        write_bot_message(get_fallback_message())

async_loop.run_forever()