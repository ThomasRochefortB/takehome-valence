import os
from langchain_core.tools import tool
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from utils import simplify_string, find_article_pubmed, get_article_pdf


@tool
def query_vector_db_articles(query: str, metadata: dict) -> list:
    """
    Retrieves relevant articles from a vector database based on a given query AND metadata filter.
    The metadata HAS to be used to filter using the article title.
    DO NOT USE THIS FUNCTION WITHOUT THE METADATA FILTER.

    Parameters:
    query (str): The search query.
    metadata (dict): Mandatory metadata to filter the search results.

    Returns:
    list: A list of query results, containing the most relevant documents.

    Example:
    query = "Why is lung cancer so hard to treat?"
    metadata = {'title': 'Lung cancer perspectives'}
    query_vector_db_articles(query, metadata)

    """

    embeddings_model = CohereEmbeddings(
        cohere_api_key=os.getenv("COHERE_API_KEY"), model="embed-multilingual-v3.0"
    )
    vectorstore = PineconeVectorStore(index_name="pubmed", embedding=embeddings_model)

    original_title = metadata["title"]
    metadata["title"] = simplify_string(metadata["title"])
    test_query = vectorstore.similarity_search(query="", k=1, filter=metadata)

    if test_query == []:
        print("No results found, Looking for the article online")

        article = find_article_pubmed(original_title)
        if article is not None:
            pdf_filetpath = get_article_pdf(article)
            loader = PyPDFLoader(pdf_filetpath)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata.update({"title": simplify_string(article.title)})
            vectorstore.from_documents(
                pages, embedding=embeddings_model, index_name="pubmed"
            )
            print("Article successfully added to the vector database.")
        else:
            print("No similar articles found on PubMed.")
            return None

    query_results = vectorstore.similarity_search(
        query,  # our search query
        k=3,  # return 3 most relevant docs
        filter=metadata,
    )
    return query_results
