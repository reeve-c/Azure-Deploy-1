import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

binary_vectors = pickle.load(open('binary_vectors.pkl', 'rb'))
unique_drinks = pickle.load(open('unique_drinks.pkl', 'rb'))
customer_data = pd.read_pickle('customer_data.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend_drinks',methods=['post'])
def recommend():

    user_name = request.form.get('user_input')
    filtered_data = customer_data[customer_data['name'] == user_name]
    drinks_list = filtered_data['product_name'].values

    drink_vectors = [binary_vectors[drink] for drink in drinks_list]

    avg_vector = np.mean(drink_vectors, axis=0)

    similarity_scores = cosine_similarity([avg_vector], list(binary_vectors.values()))[0]

    sorted_indices = np.argsort(similarity_scores)[::-1]

    recommended_drink = unique_drinks[sorted_indices]

    return render_template('index.html',data=recommended_drink[:4])


if __name__ == '__main__':
    app.run(debug=True)