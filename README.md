# takehome-valence

https://github.com/ThomasRochefortB/takehome-valence/assets/47933584/15d23556-68ac-4850-afcb-e62fc2d19c45

I wanted to show what I could do, so I went overboard with the project. I implemented a few extra features that I thought would be useful:
- I thought it was boring to retrieve from a static pdf file... so I integrated the PubMed API to chat with any OpenAccess article.
- The Assistant supports message history.
- I integrated SAFE-GPT as a tool for de novo molecule generation.
- The LangChain agent is deployed as a chat interface using chainlit.




![User Interface](public/interface.png)


Other than that, the project uses:
- Cohere Command-R-Plus as the agent
- Pinecone as the vectorstore and similarity search engine
- The Pubchem database for SMILES information
- A stacking ensemble model with SOTA performance to predict the hydration free energy


## Installation
```bash
git clone https://github.com/ThomasRochefortB/takehome-valence.git
cd takehome-valence
conda env create -f environment.yaml
conda activate takehome_valence
```

Setup your API keys in a .env file in the root directory of the project. 
```bash
COHERE_API_KEY=your_api_key
PINECONE_API_KEY=your_api_key
```

## Usage
You can go through the steps of the takehome via the working_notebook.ipybn file, but I also implemented a chat interface using chainlit.

-> To launch the chat interface, run the following command:
```bash
chainlit run app.py
```


## Hydration free energy prediction model:
* I managed to reach a 10-fold MAE that beats SOTA results from my very short lit. review and that is also under the experimental uncertainty of the freesolv dataset (+/-0.5674 kcal/mol)

* The stacking model shows a 10 fold MAE of 0.4935 which is better than the reported results in: 

[Machine learning of free energies in chemical compound space using ensemble representations: Reaching experimental uncertainty for solvation](https://pubs.aip.org/aip/jcp/article/154/13/134113/1065546) -> MAE: 0.51kcal/mol

[Machine Learning Prediction of Hydration Free Energy with Physically Inspired Descriptors](https://pubs.acs.org/doi/10.1021/acs.jpclett.2c03858?goto=supporting-info) -> RMSE: 0.8 , MAE: 0.5kcal/mol


### Process:
1. I started with simple LGBM model (with default hyperparameters) to test different feature sets. 
<center>

|           Feature set           | **10-fold CV RMSE ** | **10-fold CV MAE** |
|:-------------------------------:|:--------------------:|:------------------:|
|        Morgan Fingerprint       |         2.052        |       0.9998       |
|       RDKit's descriptors       |        1.0809        |       0.6223       |
|   SAFE-GPT's last hidden state  |        2.3724        |       1.7071       |
| SAFE-GPT's token+pos embeddings |        1.9988        |       1.3207       |
</center>


2. I then tried combining different feature sets to see if I could get a better performance than the rdkit's descriptors:
<center>

|               **Feature set**              | **10-fold CV RMSE** | **10-fold CV MAE** |
|:------------------------------------------:|:--------------------:|:------------------:|
|  Morgan Fingerprint+  RDKit's descriptors  |        1.0912        |       0.6239       |
| SAFE-GPT's token+pos + RDKit's descriptors |        1.1923        |       0.7076       |
</center>


3. I couldn't do better than the RKDit's descriptors, so I went with it and since I did not understand all of the features, I added a correlation check with the prediction labels to make sure I did not contaminate the features with the labels. I did a bit of parameter tuning to train a bunch of regressors from sklearn. I combined them using a stacking regressor with a ridge regressor at the final output:
<center>

|          **Model**          | **10-fold CV RMSE** | **10-fold CV MAE** |
|:---------------------------:|:--------------------:|:------------------:|
|             SVR             |        0.8359        |     **0.4853**     |
|              RF             |        1.1470        |       0.6764       |
|             LGBM            |        0.9810        |       0.6039       |
|      MLP (256,256,256)      |        0.9703        |       0.5970       |
| Stacking (Ridge regression) |      **0.8190**      |       0.4935       |
</center>

Well... Turns out you can reach/beat SOTA with a simple Support Vector Regression...

