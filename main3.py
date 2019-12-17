
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
#from nltk.tokenize import sent_tokenize, word_tokenize

import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import http.server
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import socketserver
from urllib.parse import urlparse, parse_qs
import cgi, cgitb

with open("intents.json") as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load('model.tflearn')

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.path = "index.html"
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)

        self.send_response(200)
        self.end_headers()
        fields = parse_qs(body)
        k = None
        for key in fields:
            k = key
        print(fields[k][0].decode("utf-8"))
        value = fields[k][0].decode("utf-8")
        
        results = model.predict([bag_of_words(value, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        #print(results)
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            value = random.choice(responses)
        else:
            value = "I didn't get that, try again."
        self.send_header("value",value)
        self.end_headers()


httpd = HTTPServer(('', 8080), MyHttpRequestHandler)
httpd.serve_forever()

