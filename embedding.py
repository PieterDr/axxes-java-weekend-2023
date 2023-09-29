import os
import pickle
import uuid

import openai as openai
import tiktoken
from tika import parser

openai.api_key = 'TODO'
# Open AI's token splitting library
tiktokenInstance = tiktoken.encoding_for_model("text-davinci-003")

# Break each document into a set of documents into an array of documents with max text length
split_pdf_documents = []


def tokenize(text):
    return tiktokenInstance.encode(text)


def detokenize(tokens):
    return tiktokenInstance.decode(tokens)


# Splits PDF text intelligently, based on sentences and tokens
def chunks_to_list(lst, n):
    new_lst = []
    for i in range(0, len(lst), n):
        new_lst.append(lst[i: i + n])
    return new_lst


def token_split(text, max_tokens=4060):
    tokens = tokenize(text)
    return [detokenize(c) for c in chunks_to_list(tokens, max_tokens)]


def token_and_sentence_split(text, max_tokens=512, window_size=0, max_sentence_tokens=700):
    raw_sentences = text.replace("\n", " ")
    raw_sentences = raw_sentences.split(".")
    raw_sentences = list(filter(None, raw_sentences))
    raw_sentences = [seq.strip() for seq in raw_sentences if seq.strip()]

    # filter(len, raw_sentences)
    # split_text_into_sentences(text, "nl")

    chunk_token_len = 0
    chunk_texts = []
    splitted = []
    sentences = []
    # make sure no sentences are longer than the max_token length
    for sentence in raw_sentences:
        if len(tokenize(sentence)) > max_sentence_tokens:
            sentences += token_split(sentence, max_tokens)
        else:
            sentences.append(sentence)

    for sentence in sentences:
        token_len = len(tokenize(sentence))
        # if adding the next sentence would exceed the max token length, add the current chunk to the list and start a new chunk
        if chunk_token_len + token_len > max_tokens:
            if chunk_texts:
                splitted.append(" ".join(chunk_texts))
            chunk_texts = []
            chunk_token_len = 0
        # otherwise, add the sentence to the current chunk
        chunk_texts.append(sentence)
        chunk_token_len += token_len
    # add the last chunk to the list
    if chunk_texts:
        splitted.append(" ".join(chunk_texts))
    # remove empty strings
    splitted = [s for s in splitted if s]
    return splitted


directory = "pdf"
split_pdf_documents_with_embeddings = [];
for file in os.listdir(directory):
    print(f"Indexing {file}")
    rawText = parser.from_file(f"{directory}/{file}")
    data = rawText['content']

    # Import uuid to generate IDs for the documents
    text_split = token_and_sentence_split(data)
    for text in text_split:
        generated_uuid = uuid.uuid4()
        uuid_str = str(generated_uuid)
        split_pdf_documents.append({
            **rawText,
            "text": text,
            "id": uuid_str
        })

    # Now, let's create some embeddings for our split PDF sentences!

    for document in split_pdf_documents:
        openAiResponse = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=document.get('text')
        )
        embedding = openAiResponse.data[0].embedding

        split_pdf_documents_with_embeddings.append({
            **document,
            "embedding": embedding
        })

tuples = []
for documents_with_embedding in split_pdf_documents_with_embeddings:
    text = documents_with_embedding["text"]
    embedding = documents_with_embedding["embedding"]
    tuples.append((text, embedding))

pickle.dump(tuples, open("db.pkl", "wb"))

print("done")
