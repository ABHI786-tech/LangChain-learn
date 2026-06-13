from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

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

prompt1 = template1.invoke({"topic": "black hole"})

result = model.invoke(prompt1)


prompt2 = template2.invoke({"text": result.content})

result1 = model.invoke(prompt2)

print(result1.content)
