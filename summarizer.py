"""Summarization and study-term extraction for Clarify.

The transformer model is loaded only when a long enough text needs it. If the
model produces a clearly repetitive result or cannot load, Clarify uses a
simple extractive fallback that selects whole sentences from the original text.
"""

from collections import Counter
import re
import threading

MODEL_NAME = "Falconsai/text_summarization"
_MODEL_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_tokenizer = None
_model = None

LENGTH_RATIOS = {
    "quick": 0.30,
    "balanced": 0.45,
    "detailed": 0.60,
}

STOPWORDS = set("""
a an the and or but if then so because while as at by for from in into of on onto to up with
about after before between during through over under again further once here there when where
why how all any both each few more most other some such no nor not only own same than too very
can could may might must shall should will would do does did doing have has had having is am are
was were be been being it its this that these those i me my we our you your he she they them their
what which who whom whose being also approximately around one two three first second used use
uses using there about complete completes completing took takes take travels traveling travel
causes cause caused called make makes made known become becomes became move moves moved moving
revolve revolves revolving text say says said include includes including example examples
revolves travels causes takes complete days means refers defined known process toward another
""".split())


def _split_sentences(text):
    """Split ordinary prose into sentences while keeping sentence punctuation."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", cleaned)
    return [piece.strip() for piece in pieces if piece.strip()]


def _content_words(text):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    return [word.strip("'-") for word in words if len(word.strip("'-")) > 2 and word.strip("'-") not in STOPWORDS]


def _extractive_summary(text, length="balanced"):
    """Pick high-value complete sentences from the original text."""
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return " ".join(sentences)

    all_words = _content_words(text)
    frequencies = Counter(all_words)
    if not frequencies:
        return " ".join(sentences[: max(1, round(len(sentences) * LENGTH_RATIOS.get(length, 0.45)))])

    scores = []
    for index, sentence in enumerate(sentences):
        words = _content_words(sentence)
        score = sum(1 + (1 / frequencies[word]) for word in words)
        # A small opening-sentence bonus often helps preserve the topic.
        score += max(0, 1.0 - index * 0.12)
        if words:
            score /= len(words) ** 0.45
        scores.append((score, index, sentence, len(re.findall(r"\b\w+\b", sentence))))

    target_ratio = LENGTH_RATIOS.get(length, 0.45)
    target_words = max(1, int(len(re.findall(r"\b\w+\b", text)) * target_ratio))
    # With very short passages, retain at least two sentences when available.
    minimum_sentences = 2 if len(sentences) <= 4 else 1

    ranked = sorted(scores, key=lambda row: row[0], reverse=True)
    chosen = []
    chosen_words = 0
    # On short passages, keep the opening sentence so the summary retains its topic.
    if len(sentences) <= 4:
        first_item = next(item for item in scores if item[1] == 0)
        chosen.append(first_item)
        chosen_words += first_item[3]
    for item in ranked:
        if item in chosen:
            continue
        if chosen_words >= target_words and len(chosen) >= minimum_sentences:
            break
        chosen.append(item)
        chosen_words += item[3]

    chosen_indices = {item[1] for item in chosen}
    return " ".join(sentence for i, sentence in enumerate(sentences) if i in chosen_indices)


def _load_model():
    """Load Hugging Face objects once, only when needed."""
    global _tokenizer, _model
    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    with _MODEL_LOCK:
        if _tokenizer is None or _model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            _model.eval()
    return _tokenizer, _model


def _is_repetitive(text):
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 16:
        return False
    if len(set(words)) / len(words) < 0.42:
        return True
    four_grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    counts = Counter(four_grams)
    return any(count > 1 for count in counts.values())


def summarize_text(text, length="balanced"):
    """Return a summary, preferring the model for longer passages."""
    text = (text or "").strip()
    if not text:
        return ""

    ratio = LENGTH_RATIOS.get(length, 0.45)
    # Tiny passages often get repeated by general-purpose models. Use whole
    # sentences for these short inputs, without downloading a large model.
    if len(re.findall(r"\b\w+\b", text)) < 70:
        return _extractive_summary(text, length)

    try:
        tokenizer, model = _load_model()
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_token_count = int(encoded["input_ids"].shape[1])

        max_new_tokens = max(24, min(180, int(input_token_count * ratio)))
        with _GENERATION_LOCK:
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_beams=4,
                do_sample=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
                early_stopping=True,
            )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if not generated or _is_repetitive(generated):
            return _extractive_summary(text, length)
        return generated
    except Exception:
        # The website can still return a usable, source-based summary if the
        # model is unavailable, cannot download, or runs out of resources.
        return _extractive_summary(text, length)


def extract_key_terms(text, limit=8):
    """Find useful-looking words and short phrases using lightweight rules."""
    sentences = _split_sentences(text)
    word_counts = Counter()
    capitalized_terms = Counter()
    phrase_counts = Counter()

    for sentence in sentences:
        raw_words = re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", sentence)
        words = [word.strip("'-") for word in raw_words]
        lowered = [word.lower() for word in words]

        for original, word in zip(words, lowered):
            normalized = word[:-2] if word.endswith("'s") else word
            if (len(normalized) >= 3 and normalized not in STOPWORDS
                    and normalized not in {"tilted", "revolve", "revolves", "takes", "travel", "travels", "causes", "complete"}
                    and normalized.isalpha()):
                word_counts[normalized] += 1
                if original[:1].isupper() and normalized not in {"the", "this", "that", "these", "those"}:
                    capitalized_terms[normalized] += 1

        # Two-word terms can be useful, e.g. "tilted axis".
        for index in range(len(lowered) - 1):
            first, second = lowered[index:index + 2]
            if (len(first) >= 3 and len(second) >= 3
                    and first not in STOPWORDS and second not in STOPWORDS
                    and "'" not in words[index] and "'" not in words[index + 1]
                    and first not in {"revolves", "takes", "travels", "causes", "complete", "means", "refers", "defined", "known"}
                    and second not in {"revolves", "takes", "travels", "causes", "complete", "means", "refers", "defined", "known"}):
                phrase_counts[f"{words[index]} {words[index + 1]}"] += 1

    candidates = {}
    for word, count in word_counts.items():
        score = count * 2 + (capitalized_terms[word] * 1.5) + min(len(word), 10) * 0.08
        candidates[word] = max(candidates.get(word, 0), score)

    for phrase, count in phrase_counts.items():
        parts = phrase.lower().split()
        # Prefer meaningful multiword phrases, but not every adjacent word pair.
        score = count * 2.1 + sum(capitalized_terms[p] for p in parts) * 0.8
        candidates[phrase] = max(candidates.get(phrase, 0), score)

    # Prefer a phrase over a single word when the phrase contains that word.
    ranked = sorted(candidates.items(), key=lambda item: (-item[1], -len(item[0])))
    chosen = []
    seen_single_words = set()
    for term, _score in ranked:
        normalized = term.lower()
        if normalized in seen_single_words:
            continue
        chosen.append(term)
        if " " not in term:
            seen_single_words.add(normalized)
        if len(chosen) >= limit:
            break

    return [term.title() if term.islower() else term for term in chosen]


def extract_definitions(text, limit=12):
    """Extract definitions explicitly written in the source text.

    This deliberately does not invent dictionary definitions that were not
    present in the user's text.
    """
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
    """Return a sentence containing a term when no explicit definition exists."""
    for sentence in _split_sentences(text):
        if re.search(r"\b" + re.escape(term) + r"\b", sentence, re.IGNORECASE):
            return sentence
    return ""


def analyze_text(text, length="balanced"):
    """Create all study outputs for the Flask page."""
    terms = extract_key_terms(text)
    definitions = extract_definitions(text)
    explicit_terms = {item["term"].lower() for item in definitions}

    term_cards = []
    for term in terms:
        context = _context_for_term(term, text)
        has_definition = any(term.lower() in known or known in term.lower() for known in explicit_terms)
        term_cards.append({
            "term": term,
            "context": context,
            "has_definition": has_definition,
        })

    return {
        "summary": summarize_text(text, length),
        "key_terms": term_cards,
        "definitions": definitions,
        "used_fallback_possible": True,
    }
