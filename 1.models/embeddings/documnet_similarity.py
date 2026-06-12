from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import numpy as np
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(models = 'text-embedding-3-large', dimensions=300)

documents = [
    "virat kohli is an indian cricketer known for his aggressive batting ",
    "MS Dhoni is the former indian captain famous for his calm demeanor and finishing skills",
    "sachin tendulkar also known as the GODof the cricket, holds many batting records"
]

query = "tell me about hte virat kohli"

doc_embedding = embedding.embed_documents(documents)

query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embedding)


index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]
print(query)

print(documents[index])
print("similarity Score is:", score) 