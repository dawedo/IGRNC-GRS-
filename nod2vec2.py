import pandas as pd
import torch
import numpy as np
import networkx as nx
from node2vec import Node2Vec
import os

def generate_user_item_graph():
    """Generate bipartite graph from Mafengwo training data"""
    
    # Load Mafengwo training data
    train_file = "./Data/Mafengwo/userRating/userRatingTrain.csv"
    
    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found. Please ensure Mafengwo dataset is in ./Data/Mafengwo/")
        return None
    
    # Read training interactions
    interactions = []
    users_set = set()
    items_set = set()
    
    print("Loading Mafengwo training data...")
    
    try:
        df = pd.read_csv(train_file)
        
        # Handle different column name formats
        if 'user_id' in df.columns:
            user_col, item_col = 'user_id', 'item_id'
        else:
            user_col, item_col = df.columns[0], df.columns[1]
        
        for _, row in df.iterrows():
            user_id = int(row[user_col])
            item_id = int(row[item_col])
            
            # Check if rating column exists
            if 'rating' in df.columns:
                rating = float(row['rating'])
                # Only use positive ratings for graph construction
                if rating > 0:
                    interactions.append((user_id, item_id))
                    users_set.add(user_id)
                    items_set.add(item_id)
            else:
                # No rating column, assume all are positive
                interactions.append((user_id, item_id))
                users_set.add(user_id)
                items_set.add(item_id)
                
    except Exception as e:
        print(f"Error reading {train_file}: {e}")
        return None
    
    print(f"Found {len(interactions)} positive interactions")
    print(f"Users: {len(users_set)}, Items: {len(items_set)}")
    
    # Create bipartite graph
    G = nx.Graph()
    
    # Add users and items as nodes with bipartite attribute
    user_nodes = [f'User_{i}' for i in users_set]
    item_nodes = [f'Item_{i}' for i in items_set]
    
    G.add_nodes_from(user_nodes, bipartite=0)
    G.add_nodes_from(item_nodes, bipartite=1)
    
    # Add edges for interactions
    edges = [(f'User_{u}', f'Item_{i}') for u, i in interactions]
    G.add_edges_from(edges)
    
    print(f"Created bipartite graph:")
    print(f"  Nodes: {len(G.nodes())} ({len(user_nodes)} users, {len(item_nodes)} items)")
    print(f"  Edges: {len(G.edges())}")
    
    return G, users_set, items_set

def train_node2vec_model(G):
    """Train Node2Vec model on the bipartite graph"""
    print("Training Node2Vec model...")
    
    # Initialize Node2Vec
    node2vec = Node2Vec(
        G, 
        dimensions=64,      # Embedding dimension
        walk_length=30,     # Length of random walks
        num_walks=200,      # Number of random walks per node
        workers=4,          # Number of parallel workers
        p=1,               # Return parameter
        q=1                # In-out parameter
    )
    
    # Train the model
    model = node2vec.fit(
        window=10,          # Context window size
        min_count=1,        # Minimum word count
        batch_words=4,      # Batch size
        sg=1,               # Skip-gram model
        hs=0,               # Hierarchical softmax
        negative=10,        # Negative sampling
        epochs=10           # Number of training epochs
    )
    
    print("Node2Vec training completed!")
    return model

def extract_embeddings(model, G, users_set, items_set):
    """Extract and organize embeddings from trained model"""
    print("Extracting embeddings...")
    
    # Get max IDs to create proper sized matrices
    max_user_id = max(users_set) if users_set else 0
    max_item_id = max(items_set) if items_set else 0
    
    print(f"Max user ID: {max_user_id}, Max item ID: {max_item_id}")
    
    # Initialize embedding matrices with random values for Mafengwo dimensions
    user_embedding_matrix = np.random.normal(0, 0.1, (max_user_id + 1, 64))
    item_embedding_matrix = np.random.normal(0, 0.1, (max_item_id + 1, 64))
    
    # Extract user embeddings
    user_count = 0
    for user_id in users_set:
        user_node = f'User_{user_id}'
        if user_node in model.wv:
            user_embedding_matrix[user_id] = model.wv[user_node]
            user_count += 1
    
    # Extract item embeddings
    item_count = 0
    for item_id in items_set:
        item_node = f'Item_{item_id}'
        if item_node in model.wv:
            item_embedding_matrix[item_id] = model.wv[item_node]
            item_count += 1
    
    print(f"Extracted embeddings for {user_count}/{len(users_set)} users")
    print(f"Extracted embeddings for {item_count}/{len(items_set)} items")
    
    # Convert to PyTorch tensors
    user_tensors = torch.tensor(user_embedding_matrix, dtype=torch.float32)
    item_tensors = torch.tensor(item_embedding_matrix, dtype=torch.float32)
    
    return user_tensors, item_tensors

def save_embeddings(user_tensors, item_tensors):
    """Save embeddings to disk"""
    print("Saving embeddings...")
    
    # Save to files
    torch.save(user_tensors, "./user_tensors_all_64.pt")
    torch.save(item_tensors, "./item_tensors_all_64.pt")
    
    print(f"Saved embeddings:")
    print(f"  User embeddings: {user_tensors.shape} -> ./user_tensors_all_64.pt")
    print(f"  Item embeddings: {item_tensors.shape} -> ./item_tensors_all_64.pt")

def main():
    """Main function to generate Node2Vec embeddings for Mafengwo"""
    print("="*60)
    print("GENERATING NODE2VEC EMBEDDINGS FOR Mafengwo")
    print("="*60)
    
    # Step 1: Generate bipartite graph
    result = generate_user_item_graph()
    if result is None:
        return
    
    G, users_set, items_set = result
    
    # Step 2: Train Node2Vec model
    model = train_node2vec_model(G)
    
    # Step 3: Extract embeddings
    user_tensors, item_tensors = extract_embeddings(model, G, users_set, items_set)
    
    # Step 4: Save embeddings
    save_embeddings(user_tensors, item_tensors)
    
    print("="*60)
    print("NODE2VEC EMBEDDING GENERATION COMPLETED!")
    print("Mafengwo embeddings ready for training.")
    print("="*60)

if __name__ == "__main__":
    main()