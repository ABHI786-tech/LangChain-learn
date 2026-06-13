import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="give the sentiment of the feedback"
    )


parser2 = PydanticOutputParser(pydantic_object=Feedback)


prompt_1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive and negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)


classifier_chain = prompt_1 | model | parser2

prompt_2 = PromptTemplate(
    template="write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)
prompt_3 = PromptTemplate(
    template="write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt_3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment"),
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": "this is a terrible smartphone"})

print(result)
