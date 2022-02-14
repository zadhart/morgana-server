from flask import Flask
from flask import request
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def send_msg():

    content = request.get_json()

    utterance = content["data"]

    inputs = tokenizer(utterance, return_tensors="pt")

    res = model.generate(**inputs)

    res_dec = str(tokenizer.decode(res[0]))

    res_dec = res_dec[4: len(res_dec) - 4]

    return res_dec


if __name__ == '__main__':
    app.run(host='0.0.0.0')