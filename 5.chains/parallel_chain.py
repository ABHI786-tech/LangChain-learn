import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel

load_dotenv()


llm_1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    # repo_id="openai/gpt-oss-120b",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)
llm_2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)

model_1 = ChatHuggingFace(llm=llm_1)
model_2 = ChatHuggingFace(llm=llm_2)

prompt1 = PromptTemplate(
    template=" Summarize the following text into one short paragraph. \n {text}",
    input_variables=["text"],
)


prompt2 = PromptTemplate(
    template="Generate a 5 short question answer from the following text  \n {text}",
    input_variables=["text"],
)


prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes-> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

pararllel_chain = RunnableParallel(
    {"notes": prompt1 | model_1 | parser, "quiz": prompt2 | model_2 | parser}
)
merge_chain = prompt3 | model_1 | parser

chain = pararllel_chain | merge_chain

text = """React is a JavaScript library for building user interfaces, particularly single-page applications, that relies on a component-based architecture where reusable pieces of code are combined to create complex UIs.  These components are defined as JavaScript functions or classes and utilize JSX, a syntax extension that allows HTML-like structures to be written within JavaScript, which is then compiled into React.createElement calls for the browser. 

Key theoretical concepts include:

Props and State: Props are read-only attributes passed from parent to child components to configure them, while State is the internal, mutable data managed within a component that determines its behavior and rendering.  In modern React, state is typically managed in functional components using Hooks like useState. 
Virtual DOM and Reconciliation: React maintains a lightweight copy of the real DOM called the Virtual DOM. When state changes, React updates this virtual representation and uses a reconciler to efficiently calculate the minimal set of changes needed to update the actual browser DOM, ensuring high performance. 
Lifecycle and Side Effects: Components go through mounting, updating, and unmounting phases. In functional components, the useEffect hook is used to handle side effects (such as data fetching or DOM manipulation) that occur after rendering, replacing the need for class-based lifecycle methods like componentDidMount. 
One-Way Data Flow: React enforces a unidirectional data flow where data moves down from parent to child via props, preventing unexpected side effects and making the application state predictable and easier to debug """
result = chain.invoke({"text": text})

print(result)
