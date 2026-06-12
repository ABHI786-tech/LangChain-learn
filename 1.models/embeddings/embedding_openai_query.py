from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Delhi is the capital of india")

print(str(result))



# for docs 

# embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

# documents = [
#     "delhi is the capital of the india ",
#     "kolkata is the capital of the west bengal"
# ]
# result = embedding.embed_documents(documents)

# print(str(result))