from pymed import PubMed
from fuzzywuzzy import fuzz
from pymed import PubMed
from fuzzywuzzy import fuzz
import requests
import os
from langchain_core.tools import tool
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore  
from langchain_community.document_loaders import PyPDFLoader

def find_article_pubmed(title: str):
    """
    Find an article on PubMed based on the given title.
    

    Parameters:
    title (str): The title of the article to search for on PubMed.

    Returns:
    [pymed.article.PubMedArticle]: The first article found on PubMed that has a similarity score above the threshold,
    or None if no matching article is found.

    """
    # Initialize PubMed API
    pubmed = PubMed(tool="MyTool", email="my@email.address")
    # Query PubMed
    results = pubmed.query(title, max_results=10)

    # Define a similarity threshold (100 is an exact match, lower values allow for more differences)
    threshold = 90

    # Iterate over results and check similarity
    for article in results:
        similarity_score = fuzz.ratio(article.title, title)
        if similarity_score >= threshold:
            print(f"Found a similar title with a similarity score of {similarity_score}:")
            print(article.title)
            return article
    
    return None

def get_article_pdf(article):
    # Your email for the API request
    email = "thomas.rochefort.beaudoin@gmail.com"

    # The DOI of the article
    doi = article.doi.split('\n')[0]

    # Unpaywall API URL
    unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

    # Send a GET request to Unpaywall API
    response = requests.get(unpaywall_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Print article metadata
        print(f"Title: {data['title']}")
        print(f"Authors: {', '.join([author['given'] + ' ' + author['family'] for author in data['z_authors']])}")
        print(f"Journal: {data['journal_name']}")
        print(f"Published Date: {data['published_date']}")

        # Check if an open access PDF URL is available
        if data['is_oa']:
            pdf_url = data['best_oa_location']['url_for_pdf']
            print(f"Open access PDF available at: {pdf_url}")

            # Create the folder if it does not exist
            folder_path = ".saved_pdfs"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Download the PDF
            pdf_response = requests.get(pdf_url)
            pdf_filename = f"{data['doi'].replace('/', '_')}.pdf"
            pdf_filepath = os.path.join(folder_path, pdf_filename)

            with open(pdf_filepath, "wb") as file:
                file.write(pdf_response.content)
            
            print(f"Downloaded the PDF as: {pdf_filepath}")
        else:
            print("No open access version available.")
    else:
        print(f"Failed to retrieve information for DOI: {doi}, Status code: {response.status_code}")
    return pdf_filepath


@tool
def query_vector_db_articles(query: str, metadata: dict) -> list:
    """
    Retrieves relevant articles from a vector database based on a given query AND metadata filter.
    The metadata HAS to be used to filter using the article title.

    Parameters:
    query (str): The search query.
    metadata (dict): Additional metadata to filter the search results.

    Returns:
    list: A list of query results, containing the most relevant documents.

    Example:
    query = "Why is lung cancer so hard to treat?"
    metadata = {"title": "article_title"}
    
    query_vector_db_articles(query, metadata)

    """
    # Check if 'title' field exists in metadata
    # if 'title' not in metadata:
    #     raise ValueError("Metadata must contain a 'title' field.")

    embeddings_model = CohereEmbeddings(cohere_api_key=os.getenv("COHERE_API_KEY"),model="embed-multilingual-v3.0")
    vectorstore = PineconeVectorStore(index_name="pubmed", embedding= embeddings_model )  
    
    test_query = vectorstore.similarity_search(query="",
                                k=1,
                                filter=metadata )
    
    if test_query == []:
        print("No results found, Looking for the article online")

        article = find_article_pubmed(metadata["title"])
        if article is not None:
            pdf_filetpath = get_article_pdf(article)
            loader = PyPDFLoader(pdf_filetpath)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata.update({"title": article.title})
            vectorstore.from_documents(pages, embedding=embeddings_model, index_name="pubmed")
            print("Article successfully added to the vector database.")
        else:
            print("No similar articles found on PubMed.")
            return None
    
    query_results = vectorstore.similarity_search(
        query,  # our search query
        k=3,  # return 3 most relevant docs
        filter=metadata
        )   
    return query_results
