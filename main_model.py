import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class DrinkRecommendationModel:
    def __init__(self):
        self.binary_vectors = None
        self.unique_drinks = None
        self.unique_ingredients = None

        data = pd.read_csv('Revised_Customer_Details_3.csv')

        unique_drinks = data['product_name'].unique()
        unique_ingredients = set()

        for ingredients in data['ingredients']:
            unique_ingredients.update(eval(ingredients))

        self.unique_drinks = unique_drinks
        self.unique_ingredients = unique_ingredients

        binary_vectors = {}

        for drink in unique_drinks:
            drink_vector = np.zeros(len(unique_ingredients))
            ingredients = eval(data[data['product_name'] == drink]['ingredients'].values[0])
            for ingredient in ingredients:
                drink_vector[list(unique_ingredients).index(ingredient)] = 1
            binary_vectors[drink] = drink_vector

        self.binary_vectors = binary_vectors

    def compute_similarity_matrix(self):
        similarity_matrix = np.zeros((len(self.unique_drinks), len(self.unique_drinks)))

        for i in range(len(self.unique_drinks)):
            for j in range(len(self.unique_drinks)):
                similarity_matrix[i, j] = cosine_similarity([self.binary_vectors[self.unique_drinks[i]]], [self.binary_vectors[self.unique_drinks[j]]])[0, 0]

        return similarity_matrix

    def recommend_new_drink(self, drinks_list):
        drink_vectors = [self.binary_vectors[drink] for drink in drinks_list]
        avg_vector = np.mean(drink_vectors, axis=0)
        similarity_scores = cosine_similarity([avg_vector], list(self.binary_vectors.values()))[0]
        sorted_indices = np.argsort(similarity_scores)[::-1]
        recommended_drink = self.unique_drinks[sorted_indices[0]]
        return recommended_drink
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)