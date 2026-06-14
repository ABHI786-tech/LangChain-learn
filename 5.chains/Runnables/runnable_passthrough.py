import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
)

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
passthrough = RunnablePassthrough()

prompt_1 = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)

prompt_2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"],
)


joke_gen_chain = RunnableSequence(prompt_1, model, parser)

parallel_Chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "Explanation": RunnableSequence(prompt_2, model, parser),
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_Chain)
result = final_chain.invoke({"topic": "cricket"})


print(result)
