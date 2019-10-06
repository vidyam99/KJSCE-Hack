from flask import Flask, request, render_template, redirect, url_for, session
import os

DATASET_FOLDER = 'dataset'


app = Flask(__name__)
app.config['DATASET_FOLDER'] = DATASET_FOLDER

from result import analyze

@app.route('/', methods=['GET', 'POST'])
def hello_world():
	if(request.method == 'GET'):
		return render_template('index.html')
	if(request.method == 'POST'):
		if('file' not in request.files):
			print('dataset not uploaded')
			return
		dataset = request.files['file']
		res = analyze(dataset)
		out = ""
		for item in res:
			out=out+(str(item)+"\n")
		return render_template('result.html', res=out)

@app.route('/about')
def about_page():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)