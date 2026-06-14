import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel, RunnableSequence

load_dotenv()

prompt_1 = PromptTemplate(
    template="Generate a twitter post about {topic}", input_variables=["topic"]
)


model = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))


parser = StrOutputParser()

prompt_2 = PromptTemplate(
    template="Generate a Linkedin post about  {topic}", input_variables=["topic"]
)


parallel_chain = RunnableParallel(
    {
        "twiiter": RunnableSequence(prompt_1, model, parser),
        " LinkedIn": RunnableSequence(prompt_2, model, parser),
    }
)

result = parallel_chain.invoke({"topic": "AI"})


print(result)
