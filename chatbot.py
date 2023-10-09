# -*- coding: utf-8 -*-
"""# Import Necessary Libraries"""

import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

"""# Initialize the Data"""

topics_responses = 'https://raw.githubusercontent.com/smgestupa/ccs311-cs41s1-streamlit-chatbot/main/content/NLP-Chatbot-Data.csv'

chatdata_df = pd.read_csv(topics_responses)
chatdata_df.head()

topics_indices = []

# Store the starting and ending indeces of rows of a topic column
for index in range(0, len(chatdata_df.columns)):
  if index % 2 != 0:
    continue

  last_index = topics_indices[-1] if len(topics_indices) > 0 else 0
  topics_indices.append(last_index + (len(chatdata_df.iloc[:, index].values.tolist())))

# Rename the first and second columns to "Topics" and "Responses" respectively
chatdata_df.rename(columns={chatdata_df.columns[0]: 'Topics', chatdata_df.columns[1]: 'Responses'}, inplace=True)

new_topics = []
new_responses = []

# Store the values of columns beyond second index into their respective arrays
for index in range(0, len(chatdata_df.columns) - 2, 2):
  new_topics.extend(chatdata_df.iloc[:, index + 2].values.tolist())
  new_responses.extend(chatdata_df.iloc[:, index + 3].values.tolist())

new_df = DataFrame({'Topics': new_topics, 'Responses': new_responses})

# Merge the topics and responses dataframe into one column
chatdata_df = chatdata_df.merge(new_df, how='outer')

# Drop the columns starting from the third index
chatdata_df = chatdata_df.drop(chatdata_df.iloc[:, 2:], axis=1)
chatdata_df

"""# Define the function for getting the most similar response"""

def get_most_similar_response(df, query, top_k=1, index=0):
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

  # Fetch the corresponding response
  most_similar_responses = df.iloc[sorted_indeces]['Responses'].values

  response = None if len(most_similar_responses) == 0 else most_similar_responses[index]
  response_index = df[df['Responses'] == response].index.item()

  return response, response_index

"""# Initialize the Chatbot prototype"""

def suggest_related_topic(df, query, response_index):
  starting_index = 0
  ending_index = topics_indices[0]

  for index in range(0, len(topics_indices) - 1):
    starting_index = topics_indices[index] - 1 if topics_indices[index] <= response_index else starting_index
    ending_index = topics_indices[index + 1] - 1 if topics_indices[index] <= response_index < topics_indices[index + 1] else ending_index

  related_rows_df = df.iloc[starting_index:(ending_index + 1), :].reset_index()

  response, _ = get_most_similar_response(related_rows_df, query, top_k=2, index=-1)
  topic = related_rows_df[related_rows_df['Responses'] == response]['Topics'].values.item()

  return topic, response

query = input('> Prompt: ')

starting_time = round((time.time() * 1000) / 1000)
while True:
  if len(query.strip()) == 0:
    print('You have exited the terminal.')
    break
  else:
    response, response_index = get_most_similar_response(chatdata_df, query)
    print(f'Answer: {response}')

    current_time = round((time.time() * 1000) / 1000)

    if (current_time - starting_time >= 30):
      topic, response = suggest_related_topic(chatdata_df, query, response_index)
      print(f'Suggested Topic: {topic} - {response}')

    starting_time = current_time
    print()

    query = input('> Prompt: ')
