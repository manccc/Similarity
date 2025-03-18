# similarity_model.py
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the pre-trained model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(text1, text2):
    """
    Calculate the semantic similarity between two text paragraphs.
    Args:
        text1 (str): First text paragraph.
        text2 (str): Second text paragraph.
    Returns:
        float: Similarity score between 0 and 1.
    """
    # Encode the text paragraphs into embeddings
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    
    # Compute cosine similarity between the embeddings
    similarity_score = util.cos_sim(embedding1, embedding2).item()
    
    return similarity_score

def process_dataset(input_file, output_file):
    """
    Process the dataset and calculate similarity scores for all pairs of texts.
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file with similarity scores.
    """
    # Load the dataset
    df = pd.read_csv('C:\\Users\\Bhushan\\Desktop\\dn\\DataNeuron_Text_Similarity.csv')
    
    # Calculate similarity scores for each pair of texts
    similarity_scores = []
    for index, row in df.iterrows():
        text1 = row['text1']
        text2 = row['text2']
        similarity_score = calculate_similarity(text1, text2)
        similarity_scores.append(similarity_score)
    
    # Add the similarity scores as a new column in the DataFrame
    df['similarity_score'] = similarity_scores
    
    # Save the results to a new CSV file
    df.to_csv(output_file, index=False)
    file_path = "C:\\Users\\Bhushan\\Desktop\\dn"
    print(f"Similarity scores calculated and saved to '{file_path}'")

# Example usage
if __name__ == "__main__":
    input_file = "dataset.csv"  # Input CSV file
    output_file = "dataset_with_scores.csv"  # Output CSV file
    process_dataset(input_file, output_file)