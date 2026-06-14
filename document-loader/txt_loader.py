from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)
model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="write a summary for the following text - \n {text}",
    input_variables=["text"],
)
parser = StrOutputParser()

loader = TextLoader("ai_doc.txt", encoding="utf-8")

docs = loader.load()

# print(docs[0])

chain = prompt | model | parser

result = chain.invoke({"text": docs[0].page_content})

print(result)
