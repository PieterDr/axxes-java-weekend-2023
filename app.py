import pickle

import numpy as np
import openai
from flask import jsonify, Flask, request

openai.api_key = ""

# Load the knowledge base and embedding vectors
db = pickle.load(open('db.pkl', 'rb'))


def vectorize_text(text):
    result = openai.Embedding.create(model='text-embedding-ada-002', input=text)
    return result.data[0].embedding


def vector_similarity(x, y):
    return np.dot(np.array(x), np.array(y))


def order_contexts_by_query_similarity(query, db):
    query_embedding = vectorize_text(query)

    similarities = sorted([
        (vector_similarity(query_embedding, embedding), text) for text, embedding in db
    ], reverse=True)

    return similarities


def find_context(query, max_items=5):
    similar_contexts = order_contexts_by_query_similarity(query, db)[:max_items]
    context = ' '.join(text for (similarity, text) in similar_contexts)
    print(context)
    return context


def answer_query(query, context):
    content = f'Beantwoord de volgende vraag, rekening houdend met de gegeven context\n\n' \
              f'Context:\n {context}\n\n' \
              f'Vraag:\n {query}'
    print(content)

    messages = [
        {'role': 'user', 'content': content}
    ]

    result = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages,
        temperature=0.5,
    )

    return result.choices[0].message.content


query = "Wat is de regeling rond overuren of meeruren?"
context = find_context(query)
response = answer_query(query, context)
print(response)

app = Flask(__name__)


@app.post("/context")
def get_countries():
    body = request.get_json(silent=True)
    query = body["query"]
    context = find_context(query)
    return jsonify({"context": context})
