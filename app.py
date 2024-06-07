import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample Data
movies = pd.DataFrame({
    'movieId': [1, 2, 3, 4, 5],
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy']
})

ratings = pd.DataFrame({
    'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 5, 2, 3, 5],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 2, 4, 4]
})

# Movie Recommendation System using Collaborative Filtering
def get_movie_recommendations(user_ratings, movies, ratings):
    # Create a pivot table
    movie_ratings = ratings.pivot(index='userId', columns='movieId', values='rating')
    
    # Calculate similarity between users
    user_similarity = cosine_similarity(movie_ratings.fillna(0))
    user_similarity_df = pd.DataFrame(user_similarity, index=movie_ratings.index, columns=movie_ratings.index)
    
    # Find similar users
    user_similarities = user_similarity_df[user_ratings['userId']]
    
    # Calculate weighted ratings
    weighted_ratings = movie_ratings.mul(user_similarities, axis=0)
    recommendation_scores = weighted_ratings.sum(axis=1) / user_similarities.sum()
    
    # Get recommendations for movies not yet rated by the user
    recommended_movies = recommendation_scores[~recommendation_scores.index.isin(user_ratings['movieId'])]
    recommended_movies = recommended_movies.sort_values(ascending=False).head(10)
    
    return movies[movies['movieId'].isin(recommended_movies.index)]

# Streamlit Interface
def main():
    st.title('Movie Recommendation System')
    
    st.header('Rate Movies')
    
    # User Input
    user_ratings = []
    for i, row in movies.iterrows():
        rating = st.slider(f"Rate {row['title']}", 0, 5, 0)
        if rating > 0:
            user_ratings.append({'userId': 999, 'movieId': row['movieId'], 'rating': rating})
    
    if st.button('Get Recommendations'):
        user_ratings_df = pd.DataFrame(user_ratings)
        recommendations = get_movie_recommendations(user_ratings_df, movies, ratings)
        
        st.header('Recommended Movies')
        for i, row in recommendations.iterrows():
            st.subheader(row['title'])
            st.text(row['genres'])
    
    st.header('Feedback')
    st.text('Did you like the recommendations?')
    if st.button('Yes'):
        st.text('Thank you for your feedback!')
    if st.button('No'):
        st.text('Sorry to hear that. We will improve our recommendations!')

if __name__ == '__main__':
    main()
