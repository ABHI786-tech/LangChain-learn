import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

prompt1 = PromptTemplate(
    template="generate a detail report on {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a 5 line summary from the following text  \n {text}",
    input_variables=["text"],
)


model = ChatHuggingFace(llm=llm)


parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "global warming"})

print(result)
# chain.get_graph().print_ascii()
