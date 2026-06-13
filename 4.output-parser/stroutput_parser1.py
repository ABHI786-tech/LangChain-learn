from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="write a detail report on {topic}", input_variables=["topic"]
)


# 2nd prompt -> Summary
template2 = PromptTemplate(
    template="write a five line summary on the folllowing text. /n  {text}",
    input_variables=["text"],
)


parser = StrOutputParser()


chain = template1 | model | parser | template2 | model | parser


result = chain.invoke({"topic": "peacock"})

print(result)
