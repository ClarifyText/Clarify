
"""Clarify: Flask web application with visible processing stages."""

import os
import threading
import time
import uuid


from io import BytesIO
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from summarizer import analyze_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

MAX_TEXT_CHARACTERS = 30_000
JOB_TTL_SECONDS = 60 * 60
_jobs = {}
_jobs_lock = threading.Lock()


def _clean_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        expired = [
            job_id for job_id, job in _jobs.items()
            if job.get("updated_at", job.get("created_at", 0)) < cutoff
        ]
        for job_id in expired:
            _jobs.pop(job_id, None)


def _set_job_stage(job_id, stage, progress=None):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return

        job["stage"] = stage
        job["updated_at"] = time.time()

        if progress is not None:
            job["progress"] = max(
                job.get("progress", 0),
                min(int(progress), 99),
            )

    app.logger.info("Clarify job %s: %s", job_id[:8], stage)


def _run_analysis_job(job_id, text, selected_length):
    try:
        _set_job_stage(job_id, "Finding key terms in your text…", 10)

        result = analyze_text(
            text,
            selected_length,
            progress_callback=lambda stage, progress: _set_job_stage(
                job_id, stage, progress
            ),
        )

        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job["result"] = result
                job["status"] = "done"
                job["stage"] = "Finished! Opening your study notes…"
                job["progress"] = 100
                job["updated_at"] = time.time()

        app.logger.info("Clarify job %s finished", job_id[:8])

    except Exception:
        app.logger.exception("Clarify job %s failed", job_id[:8])

        with _jobs_lock:
            job = _jobs.get(job_id)
            if job:
                job["status"] = "error"
                job["stage"] = "Clarify encountered a problem."
                job["error"] = (
                    "Clarify couldn't finish this request. "
                    "Try a shorter passage. If the problem continues, "
                    "check the Render logs."
                )
                job["updated_at"] = time.time()


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        input_text="",
        result=None,
        error=None,
        selected_length="balanced",
    )


@app.route("/analyze", methods=["POST"])
def start_analysis():
    text = (request.form.get("text") or "").strip()
    selected_length = request.form.get("length", "balanced")
    uploaded_file = request.files.get("text_file")

    if not text and uploaded_file and uploaded_file.filename:
        if not uploaded_file.filename.lower().endswith(".txt"):
            return jsonify(
                error="Please upload a .txt file or paste text into the box."
            ), 400

        text = uploaded_file.read().decode(
            "utf-8", errors="replace"
        ).strip()

    if not text:
        return jsonify(
            error="Paste some text or choose a .txt file first."
        ), 400

    if len(text) > MAX_TEXT_CHARACTERS:
        return jsonify(
            error=f"Please use text shorter than {MAX_TEXT_CHARACTERS:,} characters."
        ), 400

    if selected_length not in {"quick", "balanced", "detailed"}:
        selected_length = "balanced"

    _clean_old_jobs()
    job_id = uuid.uuid4().hex
    now = time.time()

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "processing",
            "stage": "Preparing your text…",
            "progress": 3,
            "input_text": text,
            "selected_length": selected_length,
            "created_at": now,
            "updated_at": now,
        }

    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, text, selected_length),
        daemon=True,
    )
    worker.start()

    return jsonify(
        job_id=job_id,
        progress_url=url_for("job_progress", job_id=job_id),
    )


@app.route("/progress/<job_id>", methods=["GET"])
def job_progress(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)

        if not job:
            return jsonify(
                error="This processing session expired. Please submit your text again."
            ), 404

        response = {
            "status": job["status"],
            "stage": job["stage"],
            "progress": job.get("progress", 0),
        }

        if job["status"] == "error":
            response["error"] = job.get(
                "error", "An unexpected error occurred."
            )

        if job["status"] == "done":
            response["result_url"] = url_for(
                "show_result", job_id=job_id
            )

        return jsonify(response)


@app.route("/result/<job_id>", methods=["GET"])
def show_result(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)

        if not job or job.get("status") != "done":
            return redirect(url_for("index"))

        input_text = job["input_text"]
        result = job["result"]
        selected_length = job["selected_length"]

    

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", app="Clarify")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
