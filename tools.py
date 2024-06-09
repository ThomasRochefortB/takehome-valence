import os
from langchain_core.tools import tool
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from utils import simplify_string, find_article_pubmed, get_article_pdf, smiles_to_feature_vector
import requests
from pickle import load
from train_model import get_config
import time

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
            print("Waiting for the database to update...")
            MAX_RETRIES = 5  # Define the maximum number of retries

            query_results = []
            retry_count = 0  # Initialize the retry counter

            while query_results == [] and retry_count < MAX_RETRIES:
                time.sleep(3)  # Wait for 3 seconds before each retry
                query_results = vectorstore.similarity_search(
                    query,  # our search query
                    k=3,  # return 3 most relevant docs
                    filter=metadata,
                )
                retry_count += 1  # Increment the retry counter

            if query_results == []:
                print(f"Failed to retrieve results after {MAX_RETRIES} retries.")
                return None  # or handle the failure case appropriately

            return query_results
        else:
            print("No similar articles found on PubMed.")
            return None
    else:
        query_results = vectorstore.similarity_search(
            query,  # our search query
            k=3,  # return 3 most relevant docs
            filter=metadata,
        )
        return query_results


@tool
def get_smiles_from_pubchem(name: str) -> str:
    """
    Retrieves the Canonical SMILES representation of a compound from PubChem based on its name.

    Args:
        name (str): The name of the compound to search for in PubChem.

    Returns:
        str or None: The Canonical SMILES representation of the compound, or None if not found.
    """
    # Search PubChem for the compound by name
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
    response = requests.get(search_url)
    
    if response.status_code == 200:
        data = response.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
            properties = data['PropertyTable']['Properties']
            if properties:
                return properties[0]['CanonicalSMILES']
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    return None

@tool
def predict_energy_from_smiles(smiles: str) -> float:
    """
    Predict the energy value from a SMILES string using the trained model.

    Parameters:
    smiles (str): The SMILES string of the molecule.

    Returns:
    float: Predicted energy value.
    """

    config = get_config()
    # Load the feature indices if feature selection was used
    try:
        with open(".saved_models/feature_indices.pkl", "rb") as f:
            feature_indices = load(f)
    except FileNotFoundError:
        feature_indices = None

    # Load the trained model
    with open(".saved_models/final_model.pkl", "rb") as f:
        model = load(f)

    # Load the scaler
    with open(".saved_models/scaler.pkl", "rb") as f:
        scaler = load(f)

    # Convert SMILES to feature vector
    X = smiles_to_feature_vector(smiles, config, scaler, feature_indices)

    # Make a prediction using the trained model
    prediction = model.predict([X])[0]
    print(f"Prediction for {smiles}: {prediction}")

    return prediction