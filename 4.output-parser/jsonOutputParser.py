from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of the fictional person /n {format_instruction} ",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = template | model | parser
final_result = chain.invoke({})
# prompt = template.format()

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

print(final_result)
