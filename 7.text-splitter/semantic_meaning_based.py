from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

text_splitter = SemanticChunker(
    HuggingFaceEmbeddings(
        model_name="google/embeddinggemma-300m",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    ),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)

Sample = """
One of the most important things I didn't understand about the world when I was a child is the degree to which the returns for performance are superlinear.virat kohli is the best barsman in our indian team they are very famous for our aggressive bating

Teachers and coaches implicitly told us the returns were linear. "You get out," I heard a thousand times, "what you put in." They meant well, but this is rarely true. If your product is only half as good as your competitor's, you don't get half as many customers. You get no customers, and you go out of business.

"""

docs = text_splitter.create_documents([Sample])
print(len(docs))
print(docs)
