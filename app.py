from flask import Flask, request, jsonify, render_template

from similarity_model import calculate_similarity

app = Flask(__name__)

@app.route('/')
def home():
    """
    Serve the HTML UI.
    """
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    """
    API endpoint to calculate semantic similarity between two text paragraphs.
    Request body: {"text1": "paragraph1", "text2": "paragraph2"}
    Response body: {"similarity_score": 0.2}
    """
    # Get the JSON data from the request
    data = request.get_json()
    
    # Extract text1 and text2 from the request
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    # Calculate the similarity score
    similarity_score = calculate_similarity(text1, text2)
    
    # Return the similarity score in the response
    return jsonify({"similarity_score": similarity_score})

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)