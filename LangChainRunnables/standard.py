import random
from abc import ABC, abstractmethod

class Runnable(ABC):

    @abstractmethod
    def invoke(input_data):
        pass 

class FakeLLM(Runnable):
    def __init__(self):
        print("LLM created")

    def invoke(self, prompt):
        response_list = [
            "Islamabad is teh capital of Pakistan",
            "PSL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]
        return {"response": random.choice(response_list)}

    def predict(self, prompt):
        response_list = [
            "Islamabad is teh capital of Pakistan",
            "PSL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]
        return {"response": random.choice(response_list)}

class FakePromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)
        
    def format(self, input_dict):
        return self.template.format(**input_dict)

class FakeLLMChain:
    def __init__(self, llm, prompt):
        self.llm = llm
        self.prompt = prompt
    
    def run(self, input_dict):
        final_prompt = self.prompt.format(input_dict)
        result = self.llm.predict(final_prompt)
        return result["response"]

class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data

llm = FakeLLM()
result = llm.predict("what is teh capital of Pakistan?")
print(result)

template = FakePromptTemplate(
    template="Write a poem about {topic}",
    input_variables=["topic"]
)

# result2 = template.format({"topic": "Pakistan"})
# print(result2)

# chain = FakeLLMChain(llm=llm, prompt=template)

# result3 = chain.run({"topic": "Pakistan"})
# print(result3)

chain = RunnableConnector([template, llm])
print(chain.invoke({"topic": "Pakistan"}))