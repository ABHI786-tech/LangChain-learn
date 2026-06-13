import os
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from typing import Optional, Literal

load_dotenv()


model = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=os.getenv("GROQ_API_KEY"))


class Review(BaseModel):
    Key_theme: list[str] = Field(
        description="write down the all key themes discussed in the review in a list"
    )
    summary: str = Field(description="A brief summary of the review")

    sentiment: Literal["pos", "neg", "neu"] = Field(
        description="Return sentiment of thr review either negative positive and neutral"
    )
    pros: Optional[list[str]] = Field(
        default=None, description="write down all te pros inside a list"
    )
    cons: Optional[list[str]] = Field(
        default=None, description="write down all te cons inside a list"
    )

    name: Optional[str] = Field(
        default=None, description="write the name of hte review"
    )


strucutred_model = model.with_structured_output(Review)


# result = strucutred_model.invoke(
#     """ The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.
#     """
# )
result = strucutred_model.invoke(
    """ I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast-whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera-the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware-why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy-not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors

reviewed by the abhi
    """
)


print(result)
