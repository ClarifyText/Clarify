import os
from gtts import gTTS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

model_name = "Falconsai/text_summarization"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/style.css")
def css():
    return send_from_directory(".", "style.css")


@app.route("/summarize", methods=["POST"])
def summarize():

    text = request.form["text"]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    summary_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=20,
        num_beams=4,
        no_repeat_ngram_size=3
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    # Turn summary into speech
    tts = gTTS(text=summary, lang="en")
    tts.save("output.mp3")

    return render_template(
        "index.html",
        summary=summary
    )


@app.route("/output.mp3")
def audio():
    return send_from_directory(".", "output.mp3")


if __name__ == "__main__":
    app.run(debug=True)
