from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# from dotenv import load_dotenv

# load_dotenv()

documents = [
    Document(page_content="Langchain helps developer build LLM Application easily."),
    Document(
        page_content="Chroma is a vector database optimized for LLM-based search."
    ),
    Document(page_content="Embedding convert tetx into high-dimentional vectors"),
]


embd_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstores = Chroma.from_documents(
    documents=documents, embedding=embd_model, collection_name="my_collection"
)

retriever = vectorstores.as_retriever(search_kwargs={"k": 2})


query = "what is langchain used for?"

# result = retriever.invoke(query)
result = vectorstores.similarity_search(query, k=2)

for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
