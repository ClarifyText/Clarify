"""Summarization and study-term extraction for Clarify.

The Transformer model is loaded once per running process. If model loading or
inference fails, the app falls back to selecting complete source sentences.
A progress callback lets the web page display which stage is taking time.
"""

from collections import Counter
import re
import threading

MODEL_NAME = "Falconsai/text_summarization"
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_tokenizer = None
_model = None

LENGTH_RATIOS = {"quick": 0.30, "balanced": 0.45, "detailed": 0.60}

STOPWORDS = set("""
a an the and or but if then so because while as at by for from in into of on onto to up with
about after before between during through over under again further once here there when where
why how all any both each few more most other some such no nor not only own same than too very
can could may might must shall should will would do does did doing have has had having is am are
was were be been being it its this that these those i me my we our you your he she they them their
what which who whom whose also approximately around one two three first second used use uses using
there about complete completes completing took takes take travels traveling travel causes cause caused
called make makes made known become becomes became move moves moved moving means refers define defined
water liquid process occurs occurring form forms formed one another example examples surface
""".split())


def _notify(callback, message, progress):
    """Call the optional progress handler without breaking the summarizer."""
    if callback:
        try:
            callback(message, progress)
        except Exception:
            # Progress reporting is useful, but must never prevent summarization.
            pass


def _split_sentences(text):
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def _content_words(text):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    cleaned = [word.strip("'-") for word in words]
    return [word for word in cleaned if len(word) > 2 and word not in STOPWORDS]


def _extractive_summary(text, length="balanced"):
    """Make a source-based summary using complete sentences from the input."""
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return " ".join(sentences)

    frequencies = Counter(_content_words(text))
    if not frequencies:
        ratio = LENGTH_RATIOS.get(length, 0.45)
        return " ".join(sentences[:max(1, round(len(sentences) * ratio))])

    scored = []
    for index, sentence in enumerate(sentences):
        words = _content_words(sentence)
        score = sum(1 + 1 / frequencies[word] for word in words)
        score += max(0, 1.0 - index * 0.12)
        if words:
            score /= len(words) ** 0.45
        word_count = len(re.findall(r"\b\w+\b", sentence))
        scored.append((score, index, sentence, word_count))

    target_ratio = LENGTH_RATIOS.get(length, 0.45)
    target_words = max(1, int(len(re.findall(r"\b\w+\b", text)) * target_ratio))
    minimum_sentences = 2 if len(sentences) <= 4 else 1
    ranked = sorted(scored, key=lambda row: row[0], reverse=True)
    chosen = []
    chosen_words = 0

    if len(sentences) <= 4:
        first = next(item for item in scored if item[1] == 0)
        chosen.append(first)
        chosen_words += first[3]

    for item in ranked:
        if item in chosen:
            continue
        if chosen_words >= target_words and len(chosen) >= minimum_sentences:
            break
        chosen.append(item)
        chosen_words += item[3]

    selected_indices = {item[1] for item in chosen}
    return " ".join(sentence for index, sentence in enumerate(sentences) if index in selected_indices)


def _load_model(progress_callback=None):
    """Load the Hugging Face tokenizer and model once, on CPU, when needed."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        _notify(progress_callback, "AI model is already loaded. Preparing your text…", 35)
        return _tokenizer, _model

    with _MODEL_LOCK:
        if _tokenizer is None or _model is None:
            _notify(
                progress_callback,
                "Loading the AI model for the first time. It may download model files, so this can be the slowest step…",
                28,
            )
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _model.eval()

    _notify(progress_callback, "AI model loaded. Preparing the text for summarization…", 48)
    return _tokenizer, _model


def _is_repetitive(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 16:
        return False
    if len(set(words)) / len(words) < 0.42:
        return True
    four_grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return any(count > 1 for count in Counter(four_grams).values())


def summarize_text(text, length="balanced", progress_callback=None):
    """Summarize text and report the current processing stage."""
    text = (text or "").strip()
    if not text:
        return ""

    word_count = len(re.findall(r"\b\w+\b", text))
    if word_count < 350 or word_count > 500:
        _notify(progress_callback, "Short passage detected. Selecting useful complete sentences…", 72)
        return _extractive_summary(text, length)

    try:
        tokenizer, model = _load_model(progress_callback)
        _notify(progress_callback, "Tokenizing your text and preparing the model input…", 53)
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        token_count = int(encoded["input_ids"].shape[1])
        ratio = LENGTH_RATIOS.get(length, 0.45)
        max_new_tokens = max(24, min(120, int(token_count * ratio)))

        _notify(
            progress_callback,
            "Generating your summary on the CPU. Long passages can take longer at this stage…",
            67,
        )
        import torch
        with _GENERATION_LOCK, torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=2,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                early_stopping=True,
            )

        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        _notify(progress_callback, "Checking the summary and preparing your study notes…", 88)
        if not generated or _is_repetitive(generated):
            _notify(progress_callback, "The AI result wasn't reliable, so Clarify is choosing complete sentences from your source instead…", 90)
            return _extractive_summary(text, length)
        return generated
    except Exception:
        _notify(
            progress_callback,
            "The AI model couldn't complete this step. Clarify is creating a source-based sentence summary instead…",
            90,
        )
        return _extractive_summary(text, length)


def extract_key_terms(text, limit=8):
    """Identify recurring and useful terms using lightweight text rules."""
    sentences = _split_sentences(text)
    word_counts = Counter()
    capitalized_counts = Counter()
    phrase_counts = Counter()

    for sentence in sentences:
        raw_words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", sentence)
        words = [word.strip("'-") for word in raw_words]
        lowered = [word.lower() for word in words]

        for original, word in zip(words, lowered):
            normalized = word[:-2] if word.endswith("'s") else word
            excluded = {
                "tilted", "revolve", "revolves", "takes", "travel", "travels",
                "causes", "complete", "means", "refers", "process", "called",
            }
            if len(normalized) >= 3 and normalized not in STOPWORDS and normalized not in excluded and normalized.isalpha():
                word_counts[normalized] += 1
                if original[:1].isupper():
                    capitalized_counts[normalized] += 1

        for index in range(len(lowered) - 1):
            first, second = lowered[index:index + 2]
            if (len(first) >= 3 and len(second) >= 3
                    and first not in STOPWORDS and second not in STOPWORDS
                    and first not in {"revolves", "takes", "travels", "causes", "complete", "means", "refers"}
                    and second not in {"revolves", "takes", "travels", "causes", "complete", "means", "refers"}):
                phrase_counts[f"{words[index]} {words[index + 1]}"] += 1

    candidates = {}
    for word, count in word_counts.items():
        candidates[word] = count * 2 + capitalized_counts[word] * 1.5 + min(len(word), 10) * 0.08
    for phrase, count in phrase_counts.items():
        parts = phrase.lower().split()
        candidates[phrase] = count * 2.1 + sum(capitalized_counts[p] for p in parts) * 0.8

    ranked = sorted(candidates.items(), key=lambda item: (-item[1], -len(item[0])))
    chosen = []
    seen = set()
    for term, _score in ranked:
        if term.lower() in seen:
            continue
        chosen.append(term.title() if term.islower() else term)
        if " " not in term:
            seen.add(term.lower())
        if len(chosen) >= limit:
            break
    return chosen


def extract_definitions(text, limit=12):
    """Find definitions explicitly stated in the user's text; do not invent them."""
    definitions = []
    seen = set()
    patterns = [
        re.compile(r"^\s*(?P<term>[A-Za-z][A-Za-z0-9' -]{0,48}?)\s+(?:is defined as|means|refers to|is known as|is called)\s+(?P<definition>.+)$", re.IGNORECASE),
        re.compile(r"^\s*(?P<term>[A-Za-z][A-Za-z0-9' -]{0,48}?)\s+(?:is|are)\s+(?:a|an|the)\s+(?P<definition>.+)$", re.IGNORECASE),
        re.compile(r"^\s*(?P<term>[A-Za-z][A-Za-z0-9' -]{0,48}?)\s*:\s*(?P<definition>.+)$"),
    ]
    for sentence in _split_sentences(text):
        for pattern in patterns:
            match = pattern.match(sentence)
            if not match:
                continue
            term = re.sub(r"\s+", " ", match.group("term")).strip(" .,:;-\t")
            definition = match.group("definition").strip(" .;\t")
            if not term or not definition or len(term.split()) > 7:
                continue
            key = term.lower()
            if key in seen or key == definition.lower():
                continue
            seen.add(key)
            definitions.append({"term": term, "definition": definition[:300]})
            break
        if len(definitions) >= limit:
            break
    return definitions


def _context_for_term(term, text):
    for sentence in _split_sentences(text):
        if re.search(r"\b" + re.escape(term) + r"\b", sentence, re.IGNORECASE):
            return sentence
    return ""


def analyze_text(text, length="balanced", progress_callback=None):
    """Prepare summary, key terms, and source-stated definitions."""
    _notify(progress_callback, "Finding key terms in your text…", 10)
    terms = extract_key_terms(text)

    _notify(progress_callback, "Checking for definitions explicitly stated in your text…", 17)
    definitions = extract_definitions(text)
    explicit_terms = {item["term"].lower() for item in definitions}

    _notify(progress_callback, "Preparing your summary…", 23)
    summary = summarize_text(text, length, progress_callback=progress_callback)

    _notify(progress_callback, "Putting the summary, terms, and definitions together…", 95)
    term_cards = []
    for term in terms:
        context = _context_for_term(term, text)
        has_definition = any(term.lower() in known or known in term.lower() for known in explicit_terms)
        term_cards.append({"term": term, "context": context, "has_definition": has_definition})

    return {
        "summary": summary,
        "key_terms": term_cards,
        "definitions": definitions,
    }
