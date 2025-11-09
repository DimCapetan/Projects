from flask import Flask, render_template, request, redirect
import csv

app = Flask(__name__) # Instantiation of the app object. The __name__ is the main file that we are running

@app.route("/") # A decorator. In this case, any time that we have /, it will return a function
def index():
    return render_template('index.html')

@app.route("/<string:page_name>")
def html_page(page_name):
    return render_template(page_name)

def write_to_csv(data):
    with open('database.csv', mode = 'a', newline = '') as database:
        email = data['email']
        subject = data['subject']
        message = data['message']
        csv_writer = csv.writer(database, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow([email, subject, message])



@app.route('/submit_form', methods=['POST', 'GET'])
def submit_form():
    if request.method == 'POST':
        try:
            data = request.form.to_dict('email')
            write_to_csv(data)
            return redirect('/thankyou.html')
        except:
            return 'Was not able to save into database'
    else:
        return 'Something went wrong, try again'