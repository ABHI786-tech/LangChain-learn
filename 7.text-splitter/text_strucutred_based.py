from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

text = """One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.
"""

splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=0)

result = splitter.split_text(text)


print(result)
print(len(result))

