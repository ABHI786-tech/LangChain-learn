from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="documents", glob="*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()

print(len(docs))
print(docs[0].metadata)

# second pdf 
print(docs[10].metadata)
 