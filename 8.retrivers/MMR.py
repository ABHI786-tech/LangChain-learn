from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

document = [
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


vectorstore = FAISS.from_documents(documents=document, embedding=embd_model)
retriever = vectorstore.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 1}
)
query = "what is langchain used for?"

result = retriever.invoke(query)


for i, doc in enumerate(result):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
