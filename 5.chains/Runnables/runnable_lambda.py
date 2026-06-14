import os
from langchain_core.prompts import PromptTemplate

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough, RunnableLambda,
)


def word_count(text):
    return len(text.split())


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
# passthrough = RunnablePassthrough()

prompt = PromptTemplate(
    template="write a joke about {topic}",
    input_variables=["topic"],
)


# runnable_word_counter = RunnableLambda(word_counter)
# result = runnable_word_counter.invoke("hi there, how are you")

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_count)}
)


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({"topic": "AI"})

final_result = """ {} \n Word count - {}""".format(
    result["joke"], result["word_count"]
)
# print(result)
print(final_result)
