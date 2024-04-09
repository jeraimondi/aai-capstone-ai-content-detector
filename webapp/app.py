# NOTE* - Code retrieved and modified from:
# https://towardsdatascience.com/building-a-machine-learning-web-application-using-flask-29fa9ea11dac

from flask import Flask, request, render_template # web framework
from customtransformer import custom_model, device, vocab, tokenizer
from customtransformer import predict_with_transformer


# declare a Flask app
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def main():
    
    # when form is submitted
    if request.method == "POST":
        
        # get input text from form
        text = request.form.get("input")

        # call function for prediction
        prediction = predict_with_transformer(model=custom_model, text=text, tokenizer=tokenizer, vocab=vocab)

        # format output
        if prediction >= 0.5:
            prediction = f'{(1-prediction) * 100:.2f}% Human versus {prediction * 100:.2f}% AI'
        else:
            prediction = f'{prediction * 100:.2f}% Human versus {(1-prediction) * 100:.2f}% AI'
        
    else:
        prediction = ""

    # render html template with prediction
    return render_template("website.html", output=prediction)


# run the app
if __name__ == '__main__':
    app.run(debug = True)