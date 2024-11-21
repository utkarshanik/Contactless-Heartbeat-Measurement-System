# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
from werkzeug import secure_filename
from gg import ge
from supportFile import get_frame

import utils
import os
import cv2
import nltk

app = Flask(__name__)

app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET', 'POST'])
def landing():
	return render_template('home.html')

@app.route('/vid', methods=['GET', 'POST'])
def g():	
	return render_template('vid.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
	return render_template('home.html')

@app.route('/info', methods=['GET', 'POST'])
def info():
	return render_template('info.html')

@app.route('/Regi', methods=['GET', 'POST'])
def Regi():
	if request.method == 'POST':
		name = request.form["name"]
		email = request.form["email"]
		num = request.form["num"]
		age = request.form["age"]
		symptoms = request.form["symptoms"]

   
		
		utils.export("data/"+name+"-symptoms.txt",name,"w")
				
		data = utils.getTrainData()

		def get_words_in_tweets(tweets):	
			all_words = []
			for (words, sentiment) in tweets:all_words.extend(words)
			return all_words

		def get_word_features(wordlist):		
		
			wordlist = nltk.FreqDist(wordlist)
			word_features = wordlist.keys()
			return word_features

		word_features = get_word_features(get_words_in_tweets(data))		
		


		def extract_features(document):		
			document_words = set(document)
			features = {}
			for word in word_features:
				#features[word.decode("utf8")] = (word in document_words)
				features[word] = (word in document_words)
			#print(features)
			return features

		allsetlength = len(data)
		print(allsetlength)		
		#training_set = nltk.classify.apply_features(extract_features, data[:allsetlength/10*8])		
		training_set = nltk.classify.apply_features(extract_features, data[:88])
		#test_set = data[allsetlength/10*8:]		
		test_set = data[88:]		
		classifier = nltk.NaiveBayesClassifier.train(training_set)			
		
		def classify(symptoms):
			return(classifier.classify(extract_features(symptoms.split())))
			
				
			
		f = open("data/"+ name+"-symptoms.txt", "r")	
			
		#print(tot,neg,pos)
		for symptoms in f:
			#tot = tot + 1
			result = classify(symptoms)
            
		return render_template('Regi.html',name=name,email=email,num=num,age=age,symptoms=symptoms,result=result)			    
	
	return render_template('Regi.html')


@app.route('/input', methods=['GET', 'POST'])
def input():	
	return render_template('input.html')


@app.route('/video', methods=['GET', 'POST'])
def video():
	return render_template('video.html')


@app.route('/video_stream')
def video_stream():
	return Response(ge(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/videoo')
def videoo():
	return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
