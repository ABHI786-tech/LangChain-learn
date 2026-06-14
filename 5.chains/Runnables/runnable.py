#  in the  we can use runnable without langchain. in this we use the invoke function in all prompt, llm, parser 
# the below code are without built in langchain functionality 




from abc import ABC, abstractmethod
import random


class Runnable(ABC):

  @abstractmethod
  def invoke(input_data):
    pass
  


class NakliLLM(Runnable):

  def __init__(self):
    print('LLM created')

  def invoke(self, prompt):

    response_list = [
        "Delhi is the capital of the india",
        "cricketis the one of the popular game in india ",
        "ai stand for artificial intelligence",
        "IPL is a cricket league"
    ]

    return {"response": random.choice(response_list) }


  def predict(self, prompt):

    response_list = [
        "Delhi is the capital of the india",
        "cricketis the one of the popular game in india ",
        "ai stand for artificial intelligence",
        "IPL is a cricket league"
    ]

    return {"response": random.choice(response_list) }
  

llm = NakliLLM()

llm.predict("what is the capital of india")

from tempfile import template
class NakliPromptTemplate(Runnable):

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables


  def invoke(self, input_dict):
    return self.template.format(**input_dict)
  
  def format(self, input_dict):
    return "this will remov in future so you can use invoke instead of format"


class RunnableConnector(Runnable):

  def __init__(self,runnable_list):
    self.runnable_list = runnable_list

  def invoke(self, input_data):
    for runnable in self.runnable_list:
        input_data = runnable.invoke(input_data)

    return input_data

template = NakliPromptTemplate(
    template="write a {length} poem about {topic}",
    input_variables=["length",'topic']
)

llm = NakliLLM()

chain = RunnableConnector([template, llm])

result = chain.invoke({"length": "short", "topic": "india"})


print(result)