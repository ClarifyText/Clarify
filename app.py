"""Flask web app for Clarify."""

import os
from flask import Flask, render_template, request

from summarizer import analyze_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB upload limit
MAX_TEXT_CHARACTERS = 30_000


@app.route("/", methods=["GET", "POST"])
def index():
    input_text = ""
    result = None
    error = None
    selected_length = "balanced"

    if request.method == "POST":
        input_text = (request.form.get("text") or "").strip()
        uploaded_file = request.files.get("text_file")
        selected_length = request.form.get("length", "balanced")

        # Use an uploaded .txt file if the text box is empty.
        if not input_text and uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename.lower()
            if not filename.endswith(".txt"):
                error = "Please upload a .txt file, or paste your text into the box."
            else:
                input_text = uploaded_file.read().decode("utf-8", errors="replace").strip()

        if not error and not input_text:
            error = "Paste some text or choose a .txt file first."
        elif not error and len(input_text) > MAX_TEXT_CHARACTERS:
            error = f"Please use text shorter than {MAX_TEXT_CHARACTERS:,} characters."
        elif not error:
            if selected_length not in {"quick", "balanced", "detailed"}:
                selected_length = "balanced"
            try:
                result = analyze_text(input_text, selected_length)
            except Exception as exc:
                # Keep technical details in the server log, not on the public page.
                app.logger.exception("Clarify could not analyze the submitted text")
                error = "Clarify could not process that text. Check the terminal for the error and try again."

    return render_template(
        "index.html",
        input_text=input_text,
        result=result,
        error=error,
        selected_length=selected_length,
    )


@app.route("/health")
def health():
    """Simple health-check route for hosting providers."""
    return {"status": "ok", "app": "Clarify"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
