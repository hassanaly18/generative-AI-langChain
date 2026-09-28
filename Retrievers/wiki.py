# from langchain_community.retrievers import WikipediaRetriever

# retr = WikipediaRetriever(top_k_results=2, lang="en")

# query = "The geopolitical history of india and Pakistan from the perspective of a chinese"

# docs = retr.invoke(query)
# print(docs)

import wikipedia
wikipedia.wikipedia.USER_AGENT = "MyLangChainApp/1.0"

from langchain_community.retrievers import WikipediaRetriever

retr = WikipediaRetriever(top_k_results=2, lang="en")

docs = retr.invoke("India Pakistan relations")
# for d in docs:
#     print(d.metadata["title"], "->", d.page_content[:200], "\n")

for i, doc in enumerate(docs):
    print(f"\n--- {i+1} ---")
    print(f"Content: \n{doc.page_content}...")