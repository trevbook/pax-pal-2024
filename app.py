import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from openai import OpenAI
import numpy as np

# Set up the OpenAI client 
client = OpenAI()

# =======
# METHODS
# =======
# Below, I'll declare some methods that'll help with this app. 

@st.cache_data
def load_data(url):
    """
    Load the data from a .parquet file located at the specified URL.
    
    Args:
    - url (str): The URL where the .parquet file is located.

    Returns:
    - pandas.DataFrame: The loaded data.
    """
    # Fetch the content from the URL
    r = requests.get(url)
    if r.status_code == 200:
        # If the request was successful, read the content into a DataFrame
        return pd.read_parquet(BytesIO(r.content))
    else:
        # If the request failed, return an empty DataFrame
        return pd.DataFrame()

def embed_text(text):
    """
    Embeds some text using the OpenAI text-embedding-3 model.
    """
    response = client.embeddings.create(input=text, model="text-embedding-3-large")
    return response.data[0].embedding

# ======
# LAYOUT
# ======
# Now, I'm going to set up the layout of the page.

# Set up the page title and header
st.title('PAX Pal 2024')

# URL of the .parquet file
url = 'https://raw.githubusercontent.com/trevbook/pax-pal-2024/main/data/pax-east-game-descriptions.parquet'

# Load the data
df = load_data(url)

# Take in the user's input
user_query = st.text_input("Describe a game you want to play...")

# If the user has entered a query, embed it
if user_query:
    query_emb = embed_text(user_query)

    # Figure out the most similar game
    sim_df = df.copy()
    sim_df["similarity"] = sim_df["description_emb"].apply(lambda x: np.dot(x, query_emb))

    # Sort the DataFrame by similarity
    sim_df = sim_df.sort_values("similarity", ascending=False)

    # For each of the top 5 most similar games, display the title and description
    for i in range(7):

        # Grab some information about the game
        title=sim_df.iloc[i]['ganme_name']
        description=sim_df.iloc[i]['description']
        publisher=sim_df.iloc[i]['publisher']
        booth=sim_df.iloc[i]['booth']
        link=sim_df.iloc[i]['link']
        similarity=sim_df.iloc[i]['similarity']

        # Create a Markdown string to display the information
        info_str = f"""## {title} \n\n**Publisher:** {publisher}\n\n**Booth:** {booth}\n\n**Description:** {description}\n\n**Similarity to query:** {similarity * 100:.2f}%\n\n"""

        # If the game has a link, add it to the Markdown string
        if link:
            info_str += f"**[Learn more]({link})**"

        # Display the information
        st.markdown(info_str)