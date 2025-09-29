from flask import Flask, render_template, request, redirect, url_for,session, flash
import docx
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from math import sqrt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from googleapiclient.discovery import build
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import os
import re
import math
import requests
import docx
import nltk
import aiohttp
import asyncio
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'b3f67e3c2a9f495b8e13b4298ed2b4767a017e935c7d4600b1ae4b5d2f902f69'

#login dummy details
users = {}

WORD = re.compile(r'\w+')

google_api_key = "AIzaSyA7v2WJdmXkmudSfaa0j8a3iVavfk7UzgM"
google_cse_id = "758ad3e78879f0e08"

def read_file(file):
    try:
        if file.filename.endswith('.docx'):
            doc = docx.Document(file)
            text = ' '.join(paragraph.text for paragraph in doc.paragraphs)
            return text
        else:
            return file.read().decode('utf-8')  # Decode for text files
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

def fetch_google_results(query, api_key, cse_id, num_results=4):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        results = service.cse().list(q=query, cx=cse_id, num=num_results).execute()
        return [item['link'] for item in results.get('items', [])]
    except Exception as e:
        print(f"Error fetching Google results: {e}")
        return []

async def fetch_url_content(url):
    """
    Fetches the HTML content of a given URL asynchronously.
    """
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    return ' '.join(soup.stripped_strings)  # Clean and return the text
                else:
                    print(f"Failed to fetch {url}: HTTP {response.status}")
                    return ""
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return ""

async def scrape_webpage_content_async(urls):
    """
    Orchestrates multiple asynchronous URL fetches.
    """
    tasks = [fetch_url_content(url) for url in urls]
    return await asyncio.gather(*tasks)  # Gather results from all tasks

def preprocess_text(text):
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    en_stops = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = word_tokenize(text)
    return " ".join(word for word in words if word not in en_stops)

def text_to_vector(text):
    """
    Converts text into a frequency vector using regex-based word extraction.
    """
    words = WORD.findall(text)
    return Counter(words)

def get_cosine(vec1, vec2):
    """
    Calculates cosine similarity between two vectors.
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[x] * vec2[x] for x in intersection)

    sum1 = sum(vec1[x]**2 for x in vec1.keys())
    sum2 = sum(vec2[x]**2 for x in vec2.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    return 0.0 if denominator == 0 else float(numerator) / denominator

def calculate_similarity(text1, text2):
    """
    Calculates cosine similarity between two texts.
    """
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()

    text1_vector = text_to_vector(text1)
    text2_vector = text_to_vector(text2)

    tfidf_similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0] * 100
    manual_cosine = get_cosine(text1_vector, text2_vector) * 100

    combined_similarity = (0.7 * tfidf_similarity) + (0.3 * manual_cosine)

    return combined_similarity

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if request.method == 'POST':
            # Extract input (file or text query)
            file_content = ""
            if request.form['choice'] == 'file' and 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return "No file selected. Please upload a file.", 400
                file_content = read_file(file)
            elif request.form['choice'] == 'text' and 'search_query' in request.form:
                file_content = request.form['search_query']
                if not file_content.strip():
                    return "No text query provided. Please enter a search query.", 400
            else:
                return "Invalid input. Please upload a file or enter a query.", 400
            
            # Preprocess input content
            preprocessed_content = preprocess_text(file_content)

            # Fetch Google search results
            search_results = fetch_google_results(preprocessed_content, google_api_key, google_cse_id)
            if not search_results:
                print("No search results returned.")
                return "No results found from Google search.", 400

            # Asynchronously scrape webpage content
            scraped_contents = asyncio.run(scrape_webpage_content_async(search_results))

            # Analyze similarity
            similarity_results = []
            for url, content in zip(search_results, scraped_contents):
                if content:
                    similarity = calculate_similarity(preprocessed_content, preprocess_text(content))
                    similarity_results.append((url, similarity))
            session['results'] = similarity_results

            flash("Analysis complete! Here are your results.")
        return redirect(url_for('show_results'))
    except Exception as e:
        print(f"Error during analysis: {e}")
        return "An internal error occurred. Please try again later.", 500
    
@app.route('/results')
def show_results():
    if 'results' in session:
        return render_template('results.html', results=session['results'])
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        # Debug: Print the users dictionary
        print(users)

        if username in users:
            flash("Username already exists. Please choose a different username.", "error")
        elif password != confirm_password:
            flash("Passwords do not match. Please try again.", "error")
        else:
            # Save user credentials
            users[username] = password
            flash("Registration successful! You can now log in.", "success")
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Debugging statements
        print(f"Attempted login: {username} with password {password}")
        print(f"Current users: {users}")

        if username in users and users[username] == password:
            session['user'] = username
            flash("Login successful!", "success")
            return redirect(url_for('index'))
        else:
            flash("Invalid username or password.", "error")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True,threaded=True)