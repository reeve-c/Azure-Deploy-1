from main_model import DrinkRecommendationModel

model = DrinkRecommendationModel()
similarity_matrix = model.compute_similarity_matrix()

drinks_list = ['Espresso', 'Cappuccino', 'Iced Coffee']
recommended_drink = model.recommend_new_drink(drinks_list)

print(f"Recommended drink based on {drinks_list}: {recommended_drink}")
model.save_model('recommendation_model.pkl')
