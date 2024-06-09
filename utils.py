from pymed import PubMed
from fuzzywuzzy import fuzz
import requests
import re
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


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
            print(
                f"Found a similar title with a similarity score of {similarity_score}:"
            )
            print(article.title)
            return article

    return None


def get_article_pdf(article):
    # Your email for the API request
    email = "thomas.rochefort.beaudoin@gmail.com"

    # The DOI of the article
    doi = article.doi.split("\n")[0]

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
        print(
            f"Authors: {', '.join([author['given'] + ' ' + author['family'] for author in data['z_authors']])}"
        )
        print(f"Journal: {data['journal_name']}")
        print(f"Published Date: {data['published_date']}")

        # Check if an open access PDF URL is available
        if data["is_oa"]:
            pdf_url = data["best_oa_location"]["url_for_pdf"]
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
        print(
            f"Failed to retrieve information for DOI: {doi}, Status code: {response.status_code}"
        )
    return pdf_filepath


def simplify_string(text: str) -> str:
    """
    Simplifies a given string by converting it to lowercase, removing punctuation,
     and removing extra whitespace.

    Args:
        text (str): The input string to be simplified.

    Returns:
        str: The simplified string.
    """
    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# Function to convert SMILES to feature vector
def smiles_to_feature_vector(smiles, config, scaler, feature_indices=None):
    # Convert SMILES to Morgan fingerprint
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=2048)
        fingerprint_array = np.array(fingerprint)
    else:
        fingerprint_array = np.zeros(2048)

    # Collect molecular descriptors if required
    if config['use_descriptors']:
        descriptor_names = [name for name, _ in Descriptors.descList]
        descriptors = {}
        for name in descriptor_names:
            if mol is not None:
                descriptors[name] = getattr(Descriptors, name)(mol)
            else:
                descriptors[name] = 0  # Or you could choose another default value
        descriptors_array = np.array([descriptors[name] for name in descriptor_names])
    else:
        descriptors_array = np.array([])

    # Combine fingerprints and descriptors if both are used
    if config['use_morgan'] and config['use_descriptors']:
        feature_vector = np.concatenate((fingerprint_array, descriptors_array))
    elif config['use_morgan']:
        feature_vector = fingerprint_array
    elif config['use_descriptors']:
        feature_vector = descriptors_array
    else:
        raise ValueError("No features selected. Please include at least Morgan fingerprints or descriptors.")

    # Scale the feature vector
    feature_vector = scaler.transform([feature_vector])[0]

    # Apply feature selection if indices are provided
    if feature_indices is not None:
        feature_vector = feature_vector[feature_indices]

    return feature_vector

