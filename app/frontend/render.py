"""
HTML builders for the NewsCred dashboard (no Streamlit import, so they can be unit-tested).

Every piece of dynamic text (API results, LLM output, retrieved passages, file names) goes through
`esc()` before it is placed in HTML. Static markup and CSS class names are fixed strings in this file.
Nothing here computes or invents results: it only formats what the backend returned.
The returned strings contain no newlines/indentation so Markdown never turns them into code blocks.
"""

import html
from typing import Any, Dict, Iterable, List, Optional

TONE_BY_STATUS = {"ok": "ok", "skipped": "neutral", "unavailable": "warn", "timeout": "warn", "error": "err"}
STATUS_TEXT = {"ok": "Available", "skipped": "Skipped", "unavailable": "Unavailable", "timeout": "Timed out", "error": "Error"}
TONE_BY_SIGNAL = {
    "insufficient_information": "neutral", "mixed_signals": "warn",
    "warning_signs_detected": "err", "few_warning_signs_detected": "ok",
}
SIGNAL_TEXT = {
    "insufficient_information": "Not enough information", "mixed_signals": "Mixed signals",
    "warning_signs_detected": "Warning signs detected", "few_warning_signs_detected": "Fewer warning signs detected",
}


def esc(value: Any) -> str:
    """HTML-escape any value (None -> empty string)."""
    return html.escape("" if value is None else str(value), quote=True)


def pill(text: str, tone: str = "neutral") -> str:
    tone = tone if tone in {"ok", "warn", "err", "neutral", "accent"} else "neutral"
    return f'<span class="nc-pill {tone}">{esc(text)}</span>'


def status_pill(status: Optional[str]) -> str:
    status = status or "error"
    return pill(STATUS_TEXT.get(status, str(status).title()), TONE_BY_STATUS.get(status, "neutral"))


def percent(value: Optional[float]) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{value * 100:.1f}%"


def note(text: Optional[str], tone: str = "") -> str:
    return f'<div class="nc-note {tone}">{esc(text)}</div>' if text else ""


def _list(items: Iterable[Any]) -> str:
    return "<ul>" + "".join(f"<li>{esc(i)}</li>" for i in items) + "</ul>"


def _card(kicker: str, title: str, status_html: str, body: str) -> str:
    return (f'<div class="nc-card"><div class="nc-card-head"><div><div class="nc-kicker">{esc(kicker)}</div>'
            f'<h3 class="nc-card-title">{esc(title)}</h3></div><div>{status_html}</div></div>{body}</div>')


# ----------------------------------------------------------------------------- page chrome
def header_html() -> str:
    return (
        '<div class="nc-header"><div class="nc-brand">News<span>Cred</span></div>'
        '<div class="nc-sub">AI-assisted news analysis and evidence exploration</div>'
        '<p class="nc-lede">Paste a headline or article and NewsCred shows what two separate tools make of it: '
        'a machine-learning classifier, and a search of your own reference documents for passages related to the '
        'statements it makes. It points out things worth checking. It does not decide what is true.</p>'
        '<p class="nc-disclaimer">Experimental student project. Results can be wrong, and nothing here is fact-checked '
        'or verified.</p></div>'
    )


def service_status_html(health: Dict[str, Any]) -> str:
    """One-line summary of GET /health (only what the API reported)."""
    comps = health.get("components", {}) if isinstance(health, dict) else {}
    labels = [("ml_model", "ML classifier"), ("rag", "Evidence retrieval")]
    parts = []
    for key, label in labels:
        status = comps.get(key, {}).get("status", "error")
        parts.append(f"<span>{esc(label)} {status_pill(status)}</span>")
    return '<div class="nc-statusline"><span>Analysis service connected</span>' + "".join(parts) + "</div>"


# ----------------------------------------------------------------------------- result cards
def input_summary_html(inp: Dict[str, Any]) -> str:
    kind = str(inp.get("input_kind", "")).replace("_", " ")
    words, chars = inp.get("word_count"), inp.get("char_count")
    bits = []
    if isinstance(words, int):
        bits.append(f"{words:,} words")
    if isinstance(chars, int):
        bits.append(f"{chars:,} characters")
    if kind:
        bits.append(f"treated as: {kind}")
    return f'<div class="nc-hint">{esc(" · ".join(bits))}</div>'


def ml_card_html(comp: Dict[str, Any]) -> str:
    title = "Machine learning prediction"
    kicker = "A · Classifier"
    status = comp.get("status")
    if status != "ok" or not isinstance(comp.get("result"), dict):
        body = note(comp.get("error") or "The ML model is not available.", "warn")
        return _card(kicker, title, status_pill(status), body)

    res = comp["result"]
    label = res.get("label") or "n/a"
    fake_p, real_p = res.get("fake_probability"), res.get("real_probability")
    conf = str(res.get("confidence_level") or "n/a").title()

    def bar(name: str, value: Optional[float], cls: str) -> str:
        width = max(0.0, min(100.0, value * 100)) if isinstance(value, (int, float)) else 0.0
        return (f'<div class="nc-bar-row"><div class="nc-bar-top"><span>{esc(name)}</span><strong>{percent(value)}</strong></div>'
                f'<div class="nc-bar {cls}"><div style="width:{width:.1f}%"></div></div></div>')

    body = (
        f'<div class="nc-label" style="margin-top:0">Predicted label</div>'
        f'<div class="nc-verdict">{esc(label)} {pill("Confidence: " + conf, "neutral")}</div>'
        f'<div class="nc-label">Model scores</div>'
        f'{bar("Score for Fake", fake_p, "")}{bar("Score for Real", real_p, "real")}'
        f'<p class="nc-caption">Scores come from a TF-IDF + Logistic Regression model trained on the ISOT news dataset. '
        f'They describe how closely the wording matches its training examples, not the chance that the story is true.</p>'
    )
    body += "".join(note(w, "warn") for w in res.get("warnings") or [])
    body += note(comp.get("notice") or "This is a model prediction, not proof of factual correctness.", "accent")
    terms = res.get("top_terms") or {}
    if terms.get("toward_fake") or terms.get("toward_real"):
        def term_list(items):
            return "<ul>" + "".join(
                f"<li>{esc(t.get('term'))} ({float(t.get('contribution', 0)):+.3f})</li>" for t in items) + "</ul>"
        body += ('<details><summary>Words that influenced the model most</summary>'
                 '<p class="nc-caption">Model internals, not facts.</p><div class="nc-grid2">'
                 f'<div><div class="nc-label">Toward Fake</div>{term_list(terms.get("toward_fake", []))}</div>'
                 f'<div><div class="nc-label">Toward Real</div>{term_list(terms.get("toward_real", []))}</div>'
                 '</div></details>')
    return _card(kicker, title, status_pill(status), body)


def _evidence_for_claim(result: Dict[str, Any]) -> str:
    head = f'<div class="nc-claim">Claim: {esc(result.get("claim"))}</div>'
    status = result.get("retrieval_status")
    if status == "evidence_retrieved" and result.get("retrieved_evidence"):
        passages = []
        for ev in result["retrieved_evidence"]:
            page = f" · page {esc(ev['page'])}" if ev.get("page") is not None else ""
            demo = " · fictional demo document" if ev.get("is_synthetic_demo") else ""
            score = ev.get("score")
            score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            passages.append(
                f'<div class="nc-passage"><p class="nc-passage-text">{esc(ev.get("text"))}</p>'
                f'<div class="nc-meta">Rank {esc(ev.get("rank"))} · source: {esc(ev.get("source"))}{page} · '
                f'passage #{esc(ev.get("chunk_id"))} · cosine similarity {score_txt}{esc(demo)}</div></div>')
        return head + "".join(passages)
    if status == "no_relevant_evidence":
        return head + note("No sufficiently relevant passage was found in the reference collection. This does not mean the "
                            "claim is true or false; the collection may simply not cover it.", "warn")
    return head + note(result.get("error") or "Retrieval was not possible for this claim.", "warn")


def evidence_card_html(comp: Dict[str, Any]) -> str:
    title, kicker, status = "Retrieved evidence", "B · Reference documents", comp.get("status")
    if status == "skipped":
        return _card(kicker, title, status_pill(status), note(comp.get("detail") or "Evidence retrieval was not run."))
    claims = comp.get("claims") or []
    if status in ("unavailable", "error", "timeout") and not claims:
        return _card(kicker, title, status_pill(status), note(comp.get("error") or "Evidence retrieval is not available.", "warn"))

    body = ""
    if comp.get("uses_synthetic_demo_documents"):
        body += note("The reference collection is fictional demonstration data written for this project. These passages are "
                      "not real-world evidence and this is not real fact-checking.", "err")
    body += "".join(note(n) for n in comp.get("notes") or [] if "FICTIONAL" not in str(n))
    if comp.get("error"):
        body += note(comp["error"], "warn")
    body += "".join(_evidence_for_claim(r) for r in claims if isinstance(r, dict))
    body += note("Similarity scores show how close the wording or topic is. They are not probabilities of truth, and a passage "
                  "may support, contradict or merely mention a claim.", "accent")
    return _card(kicker, title, status_pill(status), body)


def assessment_card_html(fa: Dict[str, Any]) -> str:
    signal = fa.get("overall_signal", "insufficient_information")
    tone = TONE_BY_SIGNAL.get(signal, "neutral")
    body = (
        f'<p><strong>{esc(fa.get("headline"))}</strong></p>'
        '<div class="nc-label">What each component said</div>' + _list(fa.get("observations") or [])
    )
    body += "".join(f'<p class="nc-caption">{esc(n)}</p>' for n in fa.get("notes") or [])
    body += ('<div class="nc-grid2"><div><div class="nc-label">What this analysis cannot tell you</div>'
             f'{_list(fa.get("cannot_conclude") or [])}</div>'
             f'<div><div class="nc-label">Suggested next steps</div>{_list(fa.get("suggested_next_steps") or [])}</div></div>')
    body += note("This summary is not a verdict and not a credibility score. The results above stay separate.", "accent")
    if fa.get("basis"):
        body += f'<p class="nc-caption">{esc(fa["basis"])}</p>'
    return _card("C · Summary", "Overall assessment", pill(SIGNAL_TEXT.get(signal, str(signal).title()), tone), body)


def limitations_html(items: Iterable[str]) -> str:
    return _list(items)


# ----------------------------------------------------------------------------- unavailable-API panel
def backend_down_html(message: str) -> str:
    return (f'<div class="nc-card"><div class="nc-kicker">Analysis service</div>'
            f'<h3 class="nc-card-title">The analysis service is not running</h3>'
            f'<p>{esc(message)}</p>'
            '<p class="nc-caption">NewsCred runs its analysis in a separate local service. Start it, then press '
            '<em>Retry connection</em>. Nothing is analysed and no results are shown until it is reachable.</p></div>')


# ----------------------------------------------------------------------------- static pages
def how_it_works_html() -> str:
    return (
        '<div class="nc-section-title">How NewsCred works</div>'
        '<p class="nc-lede">NewsCred runs two independent analyses on the text you paste. Each one is shown in its own '
        'card, and none of them overrides the others.</p>'
        '<div class="nc-grid2">'
        '<div class="nc-card"><div class="nc-kicker">Step 1</div><h3 class="nc-card-title">Classifier</h3>'
        '<p>A TF-IDF + Logistic Regression model trained on the ISOT Fake and Real News dataset gives a Fake/Real label '
        'and two scores. It recognises wording patterns from its training data; it cannot check facts.</p></div>'
        '<div class="nc-card"><div class="nc-kicker">Step 2</div><h3 class="nc-card-title">Evidence search</h3>'
        '<p>Sentences that look like checkable statements (chosen by simple rules, or the short text itself) are matched '
        'against reference documents you supply, using sentence embeddings and a FAISS index. The closest passages are '
        'shown with their source and a similarity score.</p></div></div>'
        '<div class="nc-card" style="margin-top:1rem"><h3 class="nc-card-title">Reading the results</h3><ul>'
        '<li>A <strong>Real</strong> label means the wording resembles real news in the training data. It does not mean the '
        'story is accurate.</li>'
        '<li>A <strong>similarity score</strong> measures how close a passage is to a claim. It is not a probability, and the '
        'passage may support, contradict or only mention the claim.</li>'
        '<li>The <strong>overall assessment</strong> is a cautious summary of what the components reported. It is not a '
        'verdict or a credibility score.</li>'
        '<li>Evidence comes only from the reference documents that were indexed. If they do not cover a claim, the page says '
        'so.</li></ul></div>'
    )


def about_html() -> str:
    return (
        '<div class="nc-section-title">About NewsCred</div>'
        '<div class="nc-card"><p>NewsCred is the interface of a final-year B.Tech project, <em>AI-Based Fake News Detection and '
        'News Credibility Analysis Using Machine Learning and Evidence Retrieval</em>. It is an experimental research prototype, not a '
        'fact-checking service.</p>'
        '<div class="nc-label">Built with</div><ul>'
        '<li>Python, scikit-learn (TF-IDF, Logistic Regression) and the ISOT Fake and Real News dataset</li>'
        '<li>FAISS and sentence-transformers for local evidence retrieval</li>'
        '<li>FastAPI for the analysis service and Streamlit for this interface</li></ul>'
        '<div class="nc-label">Known limitations</div><ul>'
        '<li>The classifier was trained on one dataset with known formatting patterns, so it may not generalise to other news.</li>'
        '<li>Short headlines give less reliable classifier results than full articles.</li>'
        '<li>On an external dataset (WELFake) the classifier was clearly less accurate than on its own test data; see the Evaluation tab.</li>'
        '<li>Retrieved passages are only as good as the reference documents supplied; the sample documents included with the '
        'project are fictional.</li></ul>'
        '<div class="nc-label">Privacy</div><p>Analysis runs on your own machine. Text you paste is sent only to the local '
        'analysis service configured for this app.</p></div>'
    )
