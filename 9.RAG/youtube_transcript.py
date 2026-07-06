"""
YouTube Transcript RAG System

This module demonstrates a Retrieval-Augmented Generation (RAG) system that:
1. Fetches transcripts from YouTube videos
2. Splits the transcript into manageable chunks
3. Creates embeddings and stores them in a FAISS vector store
4. Uses a retriever to find relevant content based on user queries
5. Combines retrieved context with an LLM to answer questions about the video content

The example uses a transcript from the N8N full course video by Nick Saraev.

Dependencies:
    - youtube_transcript_api: For fetching YouTube transcripts
    - langchain: For text splitting, embeddings, and RAG pipeline
    - HuggingFace: For embeddings and LLM models
"""
import os
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import (ChatHuggingFace, HuggingFaceEmbeddings,
                                   HuggingFaceEndpoint)
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()

# N8N full course video by Nick saraev
video_id = "2GZ2SNXWK-c"
try: 
  # Fetch the YouTube transcript
  youtube = YouTubeTranscriptApi()
  transcript_list = youtube.fetch(video_id, languages=(["en"]))
  # Combine all transcript chunks into a single string
  transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
  print("no caption available")


# Split transcript into chunks for embedding and retrieval
# Using RecursiveCharacterTextSplitter with 1000 character chunks and 200 character overlap
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
chunks = splitter.split_text(transcript)
print(len(chunks))


# Create embeddings using HuggingFace model
# The paraphrase-MiniLM-L3-v2 model provides efficient semantic embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
                                   )

# Create FAISS vector store from the transcript chunks
# FAISS enables fast similarity search for document retrieval
vector_store = FAISS.from_texts(chunks, embeddings)
print("FAISS created successfully")
vector_store.index_to_docstore_id
vector_store.get_by_ids(["7740f870-0758-478c-a2c0-4fd45ddc88e4"])

# Create a retriever from the vector store
# Using similarity search to find the k=2 most relevant chunks
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":2})
retriever.invoke("what is N8N")

# Initialize HuggingFace LLM with DeepSeek model for text generation
llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)

# Wrap the LLM with ChatHuggingFace for chat-based interactions
model = ChatHuggingFace(llm=llm)


# Define the prompt template for RAG
# Combines context from retrieved documents with the user's question
prompt = PromptTemplate(
    template="""
    You are a very helpful AI assistant.
    Answer the following question:
    if context is insufficient, just says you don't know
    {context}
    Question: {question}
    """,
    input_variables=["context","question"]
)


# Execute the RAG pipeline
question = "what is N8N"
# Retrieve relevant documents from the vector store
retrieved_docs = retriever.invoke(question)

# Extract the text content from retrieved documents
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

# Format the final prompt with context and question
final_prompt = prompt.invoke({"context": context_text, "question": question})

# Get the answer from the model
answer = model.invoke(final_prompt)
print(answer.content)


# Alternative implementation using LangChain's Runnable chains
# This demonstrates a more declarative approach to building the RAG pipeline
# Uncomment to use this alternative method instead of the step-by-step approach above
#
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
#
# def format_docs(retrieved_docs):
#     """Format retrieved documents into a single context string."""
#     context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#     return context_text
#
# # Create a parallel chain that retrieves context and passes through the question
# parallel_chain = RunnableParallel({
#     'context': retriever | RunnableLambda(format_docs),
#     'question': RunnablePassthrough()
# })
#
# parallel_chain.invoke('who is N8N')
#
# # Create the full RAG chain: parallel → prompt → model → output parser
# parser = StrOutputParser()
# main_chain = parallel_chain | prompt | model | parser
# main_chain.invoke('what is N8N')