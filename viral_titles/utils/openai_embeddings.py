"""
Utilities for extracting and working with OpenAI embeddings.
"""
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def get_embedding(text, model="text-embedding-ada-002", client=None):
    """Get an embedding from the OpenAI API."""
    if client is None:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def batch_get_embeddings(texts, model="text-embedding-ada-002", batch_size=100, cache_file=None):
    """
    Get embeddings for a list of texts, with optional caching.
    
    Args:
        texts: List of text strings to embed
        model: OpenAI model to use
        batch_size: Number of texts to process in each API call
        cache_file: Path to cache embeddings (if None, no caching is used)
    
    Returns:
        List of embeddings
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # Check if cache exists and load it
    cache = {}
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached embeddings from {cache_file}")
        except Exception as e:
            print(f"Error loading cache: {e}")
            cache = {}
    
    # Identify which texts need embeddings
    texts_to_embed = []
    text_indices = []
    
    for i, text in enumerate(texts):
        text = str(text).replace("\n", " ").strip()
        if text in cache:
            continue
        texts_to_embed.append(text)
        text_indices.append(i)
    
    print(f"Need to fetch {len(texts_to_embed)} embeddings, {len(texts) - len(texts_to_embed)} already cached")
    
    # Get embeddings for texts not in cache
    if texts_to_embed:
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Getting OpenAI embeddings"):
            batch_texts = texts_to_embed[i:i+batch_size]
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    model=model
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Update cache with new embeddings
                for j, emb in enumerate(batch_embeddings):
                    text = texts_to_embed[i+j]
                    cache[text] = emb
                
                # Periodically save cache
                if cache_file and (i % (batch_size * 5) == 0 or i + batch_size >= len(texts_to_embed)):
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f)
                    print(f"Saved cache with {len(cache)} entries")
                    
            except Exception as e:
                print(f"Error in batch {i//batch_size}: {e}")
                # Continue with empty embeddings for this batch
                all_embeddings.extend([[0] * 1536] * len(batch_texts))
    
    # Construct the final embeddings list
    embeddings = []
    cache_idx = 0
    api_idx = 0
    
    for i in range(len(texts)):
        text = str(texts[i]).replace("\n", " ").strip()
        if text in cache:
            embeddings.append(cache[text])
        else:
            # This should not happen if the above code ran correctly
            print(f"Warning: No embedding found for text at index {i}")
            embeddings.append([0] * 1536)  # Empty embedding as fallback
    
    return embeddings 