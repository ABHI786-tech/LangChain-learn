import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()

prompt_1 = PromptTemplate(
    template="write a joke about {topic}", input_variables=["topic"]
)


model = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))


parser = StrOutputParser()

prompt_2 = PromptTemplate(
    template="explain the following joke - {text}", input_variables=["text"]
)


chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

result = chain.invoke({"topic": "crow"})


print(result)
