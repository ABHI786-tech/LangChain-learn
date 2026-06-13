import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)
model = ChatHuggingFace(llm=llm)

# model = ChatOpenAI()


class Review(TypedDict):

    summary: Annotated[str, "A brief summary of the review"]
    # sentiment: str
    sentiment: Annotated[
        Literal["pos", "neg", "neu"],
        "Return sentiment of thr review either negative positive and neutral",
    ]
    pros: Annotated[Optional[list[str]], "write down all te pros inside a list"]
    cons: Annotated[Optional[list[str]], "write down all te cons inside a list"]


strucutred_model = model.with_structured_output(Review)


result = strucutred_model.invoke(
    """ The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
    """
)


print(result)
