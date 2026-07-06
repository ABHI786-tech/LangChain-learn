from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_result=1, lang="en")

query = "the geopolitical history of the india and pakistan from the perspective of a chinese"


docs = retriever.invoke(query)
for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"content: \n {doc.page_content}...")


# from langchain_community.retrievers import WikipediaRetriever
# import requests


# # Step 1: Test Wikipedia API directly
# def test_wikipedia_api():
#     url = "https://en.wikipedia.org/w/api.php"

#     params = {
#         "action": "query",
#         "list": "search",
#         "srsearch": "Artificial Intelligence",
#         "format": "json",
#     }

#     headers = {"User-Agent": "MyLangChainBot/1.0 (your-email@example.com)"}

#     try:
#         response = requests.get(url, params=params, headers=headers, timeout=10)

#         print("=" * 50)
#         print("API TEST")
#         print("=" * 50)
#         print("Status Code:", response.status_code)
#         print("Content-Type:", response.headers.get("Content-Type"))
#         print("Response Preview:")
#         print(response.text[:500])
#         print("=" * 50)

#         response.raise_for_status()

#         # Try parsing JSON
#         data = response.json()
#         print("✅ Wikipedia API is working")
#         return True

#     except Exception as e:
#         print("❌ Wikipedia API Error:")
#         print(e)
#         return False


# # Step 2: Use LangChain WikipediaRetriever
# def test_langchain_retriever():
#     try:
#         retriever = WikipediaRetriever(
#             lang="en", top_k_results=3, doc_content_chars_max=4000
#         )

#         docs = retriever.invoke("Artificial Intelligence")

#         print(f"\nRetrieved {len(docs)} documents\n")

#         for i, doc in enumerate(docs, start=1):
#             print(f"\n{'='*50}")
#             print(f"Document {i}")
#             print(f"{'='*50}")

#             print("Title:", doc.metadata.get("title"))

#             print("\nContent Preview:")
#             print(doc.page_content[:500])

#             print("\nMetadata:")
#             print(doc.metadata)

#     except Exception as e:
#         print("\n❌ LangChain Retriever Error")
#         print(type(e).__name__)
#         print(str(e))


# if __name__ == "__main__":
#     api_ok = test_wikipedia_api()

#     if api_ok:
#         test_langchain_retriever()
#     else:
#         print("\nSkipping retriever test because API check failed.")
