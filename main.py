from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz

app = Flask(__name__)

loaded_model = pickle.load(open("moviesKnn20.pkl", "rb"))
sparse_matrix = pickle.load(open("sparse20.pkl", "rb"))
movie_to_idx = pickle.load(open("mapper20.pkl", "rb"))

#Welcome page
@app.route('/')
def home():
    return render_template('home.html')


#Results Page
@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        fav_movie = request.form['fav']


    def fuzzy_matching(mapper, fav_movie, verbose=True):

        match_tuple = []
        
        for title, index in mapper.items():
            ratio = fuzz.ratio(title.lower(), fav_movie.lower())
            if ratio >= 60:
                match_tuple.append((title, index, ratio))
                
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('No match is found')
            return
        if verbose:
            print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
        return match_tuple[0][1]


    def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):

        model_knn.fit(data)

        index = fuzzy_matching(mapper, fav_movie, verbose=True)

        distances, indices = model_knn.kneighbors(data[index], n_neighbors=n_recommendations+1)
        
        raw_recommends = \
            sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
        
        reverse_mapper = {v: k for k, v in mapper.items()}
        recommended_list = []

        print('Recommendations for {}:'.format(fav_movie))
        for i, (idx, dist) in enumerate(raw_recommends):
            recommended_list.append(('{0}: {1}'.format(i+1, reverse_mapper[idx], dist)))
        return recommended_list

    output = make_recommendation(loaded_model, sparse_matrix, movie_to_idx, fav_movie, 20)

    return render_template('result.html', output=output)

if __name__ == "__main__":
    app.run(debug = True)