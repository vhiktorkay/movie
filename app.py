import streamlit as st
import pandas as pd
import numpy as np
import openai
import random
import os
from dotenv import load_dotenv

# Load environment variables from .env file (if running locally)
load_dotenv()

# Load OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Sample Data
movies = pd.DataFrame({
    'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'title': ['Toy Story', 'Jumanji', 'Grumpier Old Men', 'Waiting to Exhale', 'Father of the Bride Part II',
              'Heat', 'Sabrina', 'Tom and Huck', 'Sudden Death', 'GoldenEye'],
    'genres': ['Animation|Children|Comedy', 'Adventure|Children|Fantasy', 'Comedy|Romance', 'Comedy|Drama|Romance', 'Comedy',
               'Action|Crime|Drama', 'Comedy|Romance', 'Adventure|Children', 'Action', 'Action|Adventure|Thriller']
})

ratings = pd.DataFrame({
    'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'movieId': [1, 2, 3, 1, 2, 4, 1, 3, 5, 2, 3, 5],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 2, 4, 4]
})

# Fetch random top-rated movies for rating
def get_random_top_rated_movies(movies, num=5):
    top_rated_movies = movies.sample(n=num)
    return top_rated_movies

# Generate recommendations using OpenAI GPT API
def get_gpt_recommendations(user_ratings, movies):
    prompt = "Based on these movie ratings, suggest other movies the user might like:\n"
    for _, row in user_ratings.iterrows():
        movie_title = movies[movies['movieId'] == row['movieId']]['title'].values[0]
        prompt += f"{movie_title}: {row['rating']}\n"
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    recommendations = response.choices[0].text.strip().split("\n")
    return recommendations

# Streamlit Interface
def main():
    st.title('Movie Recommendation System')
    
    st.header('Rate Movies')
    
    # Fetch random movies for rating
    random_movies = get_random_top_rated_movies(movies)
    
    # User Input
    user_ratings = []
    for i, row in random_movies.iterrows():
        rating = st.slider(f"Rate {row['title']}", 0, 5, 0)
        if rating > 0:
            user_ratings.append({'userId': 999, 'movieId': row['movieId'], 'rating': rating})
    
    if st.button('Get Recommendations'):
        user_ratings_df = pd.DataFrame(user_ratings)
        recommendations = get_gpt_recommendations(user_ratings_df, movies)
        
        st.header('Recommended Movies')
        for recommendation in recommendations:
            st.subheader(recommendation)
    
    st.header('Feedback')
    st.text('Did you like the recommendations?')
    if st.button('Yes'):
        st.text('Thank you for your feedback!')
    if st.button('No'):
        st.text('Sorry to hear that. We will improve our recommendations!')

if __name__ == '__main__':
    main()
