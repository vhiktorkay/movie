import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    
    # Add new user ratings to the movie ratings DataFrame
    for i, row in user_ratings.iterrows():
        movie_ratings.loc[row['userId'], row['movieId']] = row['rating']
    
    # Calculate similarity between users
    user_similarity = cosine_similarity(movie_ratings.fillna(0))
    user_similarity_df = pd.DataFrame(user_similarity, index=movie_ratings.index, columns=movie_ratings.index)
    
    # Get the similarities for the new user
    new_user_similarities = user_similarity_df.loc[999]
    
    # Calculate weighted ratings
    weighted_ratings = movie_ratings.T.dot(new_user_similarities) / np.array([np.abs(new_user_similarities).sum(axis=0)])
    
    # Exclude movies already rated by the new user
    rated_movies = user_ratings['movieId'].values
    recommendations = weighted_ratings[~weighted_ratings.index.isin(rated_movies)]
    recommendations = recommendations.sort_values(ascending=False).head(10)
    
    return movies[movies['movieId'].isin(recommendations.index)]

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

