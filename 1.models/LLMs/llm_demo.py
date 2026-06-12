# from langchain_openai import OpenAI
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# llm = OpenAI(model= "gpt-4o-mini")
llm = GoogleGenerativeAI(model= "gemini-2.5-flash")

result = llm.invoke("What is the capital of the India?")

print(result)
