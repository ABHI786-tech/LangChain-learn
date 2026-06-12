from sklearn.metrics.pairwise import cosine_similarity
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from dotenv import load_dotenv

# Environment variables load karein
load_dotenv()

# Google Gemini Embedding initialize karein
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview",
    # output_dimensionality=32

)

# result=embedding.embed_query("haseen is good boy")
# print(len(str(result)))

documents = [
    "virat kohli is an indian cricketer known for his aggressive batting ",
    "MS Dhoni is the former indian captain famous for his calm demeanor and finishing skills",
    "sachin tendulkar also known as the GODof the cricket, holds many batting records"
]

# query = "tell me about hte virat kohli"

# Embeddings generate karein
# doc_embedding = embedding.embed_documents(documents)
# query_embedding = embedding.embed_query(query)

# Cosine similarity calculate karein (Yeh [[score1, score2, score3]] return karta hai)
# scores = cosine_similarity([query_embedding], doc_embedding)

# Sahi sorting tarika: scores[0] ka use karke array ko flatten kiya
# index, score = sorted(list(enumerate(scores[0])), key=lambda x: x[1])[-1]

# print("Query:", query)
# print("Matched Document:", documents[index])
# print("Similarity Score is:", score)
