import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda,
    RunnableBranch,
)

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt_1 = PromptTemplate(
    template="write a detail report on {topic}",
    input_variables=["topic"],
)


prompt_2 = PromptTemplate(
    template="summarize the following text {text}",
    input_variables=["text"],
)


report_gen_chain = RunnableSequence(prompt_1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt_2, model, parser)),
    RunnablePassthrough(),
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({"topic": "Ai"})

print(result)
