import random

class FakeLLM:
    def __init__(self):
        print("LLM created")

    def predict(self, prompt):
        response_list = [
            "Islamabad is teh capital of Pakistan",
            "PSL is a cricket league",
            "AI stands for Artificial Intelligence"
        ]
        return {"response": random.choice(response_list)}

class FakePromptTemplate:
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
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


llm = FakeLLM()
result = llm.predict("what is teh capital of Pakistan?")
print(result)

template = FakePromptTemplate(
    template="Write a poem about {topic}",
    input_variables=["topic"]
)

result2 = template.format({"topic": "Pakistan"})
print(result2)

chain = FakeLLMChain(llm=llm, prompt=template)

result3 = chain.run({"topic": "Pakistan"})
print(result3)