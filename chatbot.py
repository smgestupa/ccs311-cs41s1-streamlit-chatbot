import time
import random
import asyncio
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    # If the similarity score is less than 60%
    similarity_score = similarity_scores[0][similarity_scores.argsort()[0][::-1][:top_k]] * 100

    if similarity_score < 60.0:
        unknown_message = [
            'I don\'t understand your message, please try again.',
            'I need more information about what you want to know. Keep going!',
            'Hmmm... I may have limited information about your message, I am sorry.',
            'I don\'t know what you want to say. Apologies.'
        ]

        return random.choice(unknown_message)

    # Fetch the corresponding response
    most_similar_responses = df.iloc[sorted_indeces]['Responses'].values

    response = None if len(most_similar_responses) == 0 else most_similar_responses[index]
    response_index = df[df['Responses'] == response].index.item()

    return response

def suggest_topic(df, query):
    write_bot_message(f'Did you know? {get_most_similar_response(df, query, top_k=20, index=random.randint(2, 19))}')

topics_responses = 'https://raw.githubusercontent.com/smgestupa/ccs311-cs41s1-streamlit-chatbot/main/content/NLP-Chatbot-Data.csv'

chatdata_df = pd.read_csv(topics_responses)

"""# Ask about BLEACH カテゴリー"""

"""Bleach (stylized in all caps) is a Japanese anime television series based on Tite Kubo's original manga series of the same name. This chatbot is free for you to ask anything Bleach-related, whether you're a fan or not, I hope the chatbot will provide you with useful information."""

"""Please feel free to try this and hope you enjoy! 😊"""

last_query = ''

async def every(__seconds: float, func, *args, **kwargs):
    while True:
        await asyncio.sleep(__seconds)
        func(*args, **kwargs)

async_loop = asyncio.new_event_loop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.session_state.messages.append({'role': 'assistant', 'content': get_most_similar_response(chatdata_df, "Help.")})

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

prompt = st.chat_input('Ask away!')

suggest_topic_loop = None

if prompt is not None:
    last_query = prompt

    with st.chat_message('user'):
        st.markdown(prompt)
    
    if suggest_topic_loop != None:
        suggest_topic_loop.cancel()

    suggest_topic_loop = async_loop.create_task(every(15, suggest_topic, chatdata_df, last_query))

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    write_bot_message(get_most_similar_response(chatdata_df, prompt))

async_loop.run_forever()