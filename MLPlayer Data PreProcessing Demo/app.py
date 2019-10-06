from flask import Flask, request, render_template, redirect, url_for, session
import os

DATASET_FOLDER = 'dataset'


app = Flask(__name__)
app.config['DATASET_FOLDER'] = DATASET_FOLDER
app.config['SECRET_KEY'] = 'the random string'

from inference import initialisation, validation, missing_values, drop_nonuni_col, visualisation, drop_useless, correlation, upper_case

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('dataset not uploaded')
            return
        print(request.files['file'])
        dataset = request.files['file']
        df = initialisation(dataset)
        df = upper_case(df)
        df = validation(df)
        df = drop_nonuni_col(df)
        #if(request.form['submit-button']=='Next'):
        return redirect(url_for('one', df=df))
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/one', methods=['GET','POST'])
def one():
    if request.method == 'GET':
        df = request.args.get("df");
        df = missing_values(df)
        if(request.form['submit-button']=='Next'):
            return redirect(url_for('two', df=df))
        return render_template('preprocessing-1.html')

@app.route('/two', methods=['GET','POST'])
def two():
    if(request.method=='GET'):
        df = session.get("df")
        
        return render_template('preprocessing-2.html',df=df)
    if(request.method=='POST'):
        if(request.form['next']):
            return redirect(url_for('three'))

@app.route('/three', methods=['GET','POST'])
def three():
    if(request.method=='GET'):
        df = session.get("df")
        df = upper_case(df)
        df = validation(df)
        df = drop_nonuni_col(df)
        return render_template('preprocessing-3.html',df=df)
    if(request.method=='POST'):
        if(request.form['next']):
            return redirect(url_for('four'))

@app.route('/four')
def four():
    if(request.method=='GET'):
        df = session.get("df")
        return render_template('preprocessing-4.html',df=df)

if __name__ == '__main__':
    app.run(debug=True)