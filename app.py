import requests
from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# TMDB API setup
TMDB_API_KEY = 'ffaee98e01700af9c0e0ffb837060ce7'  # Replace with your TMDB API key
TMDB_BASE_URL = 'https://api.themoviedb.org/3'
TMDB_IMAGE_URL = 'https://image.tmdb.org/t/p/w500'

# Load the dataset
data = pd.read_csv('C:\\Users\\Asus\\ML\\MovieWebsite\\dataset\\main_data.csv')
data.fillna('', inplace=True)  # Handle missing values

# Use the 'comb' column directly (assuming this column contains a combination of useful info like title and description)
cv = CountVectorizer()
count_matrix = cv.fit_transform(data['comb'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

def get_tmdb_movie_details(movie_title):
    """Get movie details (overview, authors, rating) from TMDB API"""
    search_url = f'{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_title}'
    response = requests.get(search_url)
    data = response.json()

    if data['results']:
        movie_id = data['results'][0]['id']
        
        # Fetch detailed movie info
        movie_details_url = f'{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}'
        movie_details_response = requests.get(movie_details_url)
        movie_details = movie_details_response.json()

        # Extract required details
        overview = movie_details.get('overview', 'No overview available')
        rating = movie_details.get('vote_average', 'No rating available')

        # Fetch authors (Directors and Writers)
        credits_url = f'{TMDB_BASE_URL}/movie/{movie_id}/credits?api_key={TMDB_API_KEY}'
        credits_response = requests.get(credits_url)
        credits = credits_response.json()

        directors = [crew['name'] for crew in credits.get('crew', []) if crew['job'] == 'Director']
        writers = [crew['name'] for crew in credits.get('crew', []) if crew['job'] == 'Writer']

        authors_list = ', '.join(directors + writers) if directors or writers else 'No authors listed'

        return overview, authors_list, rating
    return None, None, None

def get_movie_poster_url(movie_title):
    """Fetch the poster URL from TMDB API using the movie title"""
    search_url = f'{TMDB_BASE_URL}/search/movie?api_key={TMDB_API_KEY}&query={movie_title}'
    response = requests.get(search_url)
    data = response.json()

    if data['results']:
        movie_id = data['results'][0]['id']
        poster_path = data['results'][0].get('poster_path', None)
        if poster_path:
            return TMDB_IMAGE_URL + poster_path
    return None

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        movie_name = request.form.get('movie_name', '').strip().lower()

        # Find the closest match for the movie
        data['search_key'] = data['movie_title'].str.lower() + " " + data['director_name'].str.lower()
        match = data[data['search_key'].str.contains(movie_name, case=False, na=False)]

        if match.empty:
            return render_template("recommend.html", recommendations=[], message="Movie not found!")

        # Take the first match's index for recommendations
        movie_index = match.index[0]
        movie_info = data.iloc[movie_index]
        movie_title = movie_info['movie_title']

        # Fetch movie details from TMDB API
        overview, authors_list, rating = get_tmdb_movie_details(movie_title)

        # Get movie poster URL
        movie_poster_url = get_movie_poster_url(movie_title)

        # Get similarity scores for the movie
        similarity_scores = list(enumerate(cosine_sim[movie_index]))

        # Sort the movies by similarity score
        sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the top 10 similar movies (excluding the input movie itself)
        top_movies = []
        for i in sorted_movies[1:13]:  # Exclude the first one (which is the movie itself)
            movie_info = data.iloc[i[0]]
            movie_image = get_movie_poster_url(movie_info['movie_title'])  # Get movie poster image from TMDB
            top_movies.append({
                'title': movie_info['movie_title'],
                'director': movie_info['director_name'],
                'image': movie_image
            })

        return render_template("recommend.html", recommendations=top_movies, 
                               message=f"Recommendations for '{movie_title}'", 
                               overview=overview, authors_list=authors_list, 
                               rating=rating, movie_poster_url=movie_poster_url)

    return render_template("recommend.html", recommendations=[], message="")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.debug = True
    app.run()
