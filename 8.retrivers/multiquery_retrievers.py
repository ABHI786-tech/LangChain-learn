import os
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.multi_query import MultiQueryRetriever



# =====================================================================
# STEP 2: Initialize Mock Vector Database
# =====================================================================
print("Initializing Vector Database...")
sample_docs = [
    Document(page_content="Our enterprise change management policy mandates a 14-day review period for all software infrastructure adjustments.", metadata={"source": "policy_doc.txt"}),
    Document(page_content="Project timelines are heavily impacted by unexpected scope creep and bureaucratic approval bottlenecks.", metadata={"source": "project_management_guide.txt"}),
    Document(page_content="When a modification request is approved, the project schedule baseline must be updated within 24 hours.", metadata={"source": "operations_manual.txt"}),
]

embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(sample_docs, embeddings)
base_retriever = vector_db.as_retriever(search_kwargs={"k": 2})


# =====================================================================
# STEP 3: Create Custom Query Generation Prompt & Parser
# =====================================================================
class LineSplitOutputParser(BaseOutputParser[List[str]]):
    """Parses the LLM output into a clean list of queries split by newlines."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        # Filter out empty lines or markdown numbering if the LLM accidentally includes it
        cleaned_lines = [
            line.lstrip("0123456789.- ").strip() 
            for line in lines 
            if line.strip()
        ]
        return cleaned_lines

output_parser = LineSplitOutputParser()

# Explicitly instruct the model to think from alternative keyword angles
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate 3 
different versions of the given user question to retrieve relevant documents from a vector database. 
By generating multiple perspectives on the user question, your goal is to help the user overcome 
some of the limitations of distance-based similarity search.

Provide these alternative questions separated by newlines. Do not add introductory text, 
numbers, bullet points, or concluding notes. Just output the questions.

Original question: {question}""",
)


# =====================================================================
# STEP 4: Build the MultiQuery Chain
# =====================================================================
llm = ChatHuggingFace(temperature=0.2, model="gpt-4o-mini")

# Chain responsible solely for generating queries
query_generator_chain = QUERY_PROMPT | llm | output_parser


# =====================================================================
# STEP 5: Assemble and Run the MultiQuery Retriever
# =====================================================================
# Constructing using the custom parser/generator chain
advanced_retriever = MultiQueryRetriever(
    retriever=base_retriever, 
    llm_chain=query_generator_chain, 
    parser_key="lines" # internal mapping key for LangChain parsing logic
)

# Test query
user_query = "How do change management processes affect project timelines?"
print(f"\n--- Original User Query ---\n> {user_query}\n")

# Run the system
retrieved_documents = advanced_retriever.invoke(user_query)

print("--- Retrieved Unique Documents ---")
for idx, doc in enumerate(retrieved_documents):
    print(f"\nDocument {idx+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")