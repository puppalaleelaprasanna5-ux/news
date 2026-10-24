
"""
NewsCred - Streamlit dashboard.

Start the API first, then run Streamlit from the project root:
    python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
    python -m streamlit run app/frontend/app.py

The frontend displays results returned by the backend. It does not invent
results, scores, or citations, or replace failed API calls with demo data.
"""

import sys
from pathlib import Path

# Add the project and frontend directories before importing local modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = Path(__file__).resolve().parent

for directory in (str(FRONTEND_DIR), str(PROJECT_ROOT)):
    if directory not in sys.path:
        sys.path.insert(0, directory)

import streamlit as st

# Import sibling frontend modules directly to avoid the "app is not a package" conflict.
import render
import results_view

from helpers import (
    ANALYZE_TIMEOUT,
    API_START_COMMAND,
    BACKEND_URL,
    HEALTH_TIMEOUT,
    KIND_UNREACHABLE,
    call_api_ex,
    parse_manual_claims,
)

MAX_CHARS = 20000
PLACEHOLDER_CHOICE = "Choose an example…"

EXAMPLES = {
    "Short headline (fictional topic)": (
        "The Alder River levee in Brightwater was completed in 1990."
    ),
    "Claim the demo documents do not cover": (
        "Scientists discovered a new species of deep-sea octopus."
    ),
    "News-style paragraph (written for this demo)": (
        "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a spending "
        "bill that funds the government through September, officials said, "
        "after weeks of negotiations between the two parties over defense "
        "and health programs. The measure now goes to the House of "
        "Representatives, which is expected to vote on it later this week."
    ),
}

st.set_page_config(page_title="NewsCred", page_icon="📰", layout="wide")

st.markdown(
    f"<style>{(FRONTEND_DIR / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)


def html(markup: str) -> None:
    """Render HTML produced by the frontend rendering helpers."""
    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------

def load_example() -> None:
    choice = st.session_state.get("example_choice")
    if choice in EXAMPLES:
        st.session_state["input_text"] = EXAMPLES[choice]
        st.session_state["input_error"] = None


def refresh_health() -> None:
    """Check whether the analysis API is available."""
    data, error, kind = call_api_ex(
        "GET", "/health", timeout=HEALTH_TIMEOUT
    )
    st.session_state.update(
        health=data,
        health_error=error,
        health_kind=kind,
        health_checked=True,
    )


def retry_connection() -> None:
    st.session_state["health_checked"] = False
    st.session_state["result_error"] = None
    st.session_state["result_error_kind"] = None


def show_connection_status() -> None:
    if not st.session_state.get("health_checked"):
        with st.spinner("Connecting to the analysis service…"):
            refresh_health()

    health, error, kind = (
        st.session_state.get(key)
        for key in ("health", "health_error", "health_kind")
    )

    if health:
        html(render.service_status_html(health))
        return

    if kind == KIND_UNREACHABLE:
        html(
            render.backend_down_html(
                "NewsCred could not connect to its analysis service."
            )
        )
    else:
        html(
            render.backend_down_html(
                error or "The analysis service did not respond as expected."
            )
        )

    st.button("Retry connection", on_click=retry_connection, key="retry_top")

    with st.expander("Troubleshooting details"):
        st.markdown(f"**Address the app is trying:** `{BACKEND_URL}`")
        st.markdown(
            "**Start the service** from the project folder, in a second terminal:"
        )
        st.code(API_START_COMMAND, language="bash")
        st.markdown(
            "If you use a different host or port, update BACKEND_URL in your "
            "environment configuration and restart Streamlit. You can check "
            "the service by opening `/health` at that address."
        )
        if error:
            st.markdown("**Technical message:**")
            st.code(error, language="text")


def run_analysis(
    text: str,
    run_rag: bool,
    top_k: int,
    manual_raw: str,
) -> None:
    payload = {
        "text": text,
        "run_rag": run_rag,
        "top_k": top_k,
        "claims": parse_manual_claims(manual_raw) or None,
    }

    with st.spinner(
        "Analysing… the first request can take a while while the models load."
    ):
        data, error, kind = call_api_ex(
            "POST",
            "/analyze",
            payload,
            timeout=ANALYZE_TIMEOUT,
        )

    st.session_state["result"] = data
    st.session_state["result_error"] = error
    st.session_state["result_error_kind"] = kind

    if kind == KIND_UNREACHABLE:
        st.session_state.update(
            health=None,
            health_error=error,
            health_kind=kind,
            health_checked=True,
        )


def show_results(result: dict) -> None:
    html('<div class="nc-section-title">Results</div>')
    html(render.input_summary_html(result.get("input", {})))

    if result.get("partial"):
        html(
            render.note(
                "Some components were unavailable, so this is a partial result. "
                "The affected cards explain why.",
                "warn",
            )
        )

    components = result.get("components", {})
    html(render.ml_card_html(components.get("ml_prediction", {})))
    html(render.evidence_card_html(components.get("evidence", {})))
    html(render.assessment_card_html(result.get("final_assessment", {})))

    if result.get("warnings"):
        with st.expander("Technical notes and warnings"):
            for warning in result["warnings"]:
                st.markdown(f"- {render.esc(warning)}")

    if result.get("limitations"):
        with st.expander("Known limitations"):
            html(render.limitations_html(result["limitations"]))

    if result.get("disclaimer"):
        st.caption(result["disclaimer"])


# ---------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------

html(render.header_html())

tab_analyze, tab_how, tab_eval, tab_about = st.tabs(
    ["Analyze News", "How It Works", "Evaluation", "About"]
)

with tab_analyze:
    show_connection_status()

    st.session_state.setdefault("input_text", "")

    st.text_area(
        "Paste a news headline or article",
        key="input_text",
        height=190,
        max_chars=MAX_CHARS,
        placeholder="Paste a headline or the full text of an article here…",
        help="Up to 20,000 characters.",
    )

    st.markdown(
        '<div class="nc-hint">Full articles (100+ words) give the most useful '
        "results. Very short headlines are harder for the classifier to judge. "
        "Up to 20,000 characters.</div>",
        unsafe_allow_html=True,
    )

    st.selectbox(
        "Or try an example",
        [PLACEHOLDER_CHOICE, *EXAMPLES],
        key="example_choice",
        on_change=load_example,
    )

    with st.expander("Advanced options"):
        run_rag = st.checkbox(
            "Retrieve evidence passages",
            value=True,
            help=(
                "Searches your indexed reference documents for statements "
                "from the text."
            ),
        )

        top_k = st.slider("Passages per claim", 1, 5, 3)

        manual_raw = st.text_area(
            "Claims to search for (optional, one per line)",
            height=80,
            help=(
                "If left empty, statements are picked from the text "
                "by simple rules."
            ),
        )

    if st.button("Analyze News", type="primary"):
        if not st.session_state["input_text"].strip():
            st.session_state["input_error"] = (
                "Please paste a headline or article first."
            )
        else:
            st.session_state["input_error"] = None
            run_analysis(
                st.session_state["input_text"],
                run_rag,
                top_k,
                manual_raw,
            )

    if st.session_state.get("input_error"):
        st.error(st.session_state["input_error"])

    if st.session_state.get("result_error"):
        if st.session_state.get("result_error_kind") == KIND_UNREACHABLE:
            st.error(
                "The analysis could not run because the analysis service "
                "is not reachable. See the troubleshooting details above, "
                "then press Retry connection."
            )
        else:
            st.error(st.session_state["result_error"])

    if st.session_state.get("result"):
        show_results(st.session_state["result"])


with tab_how:
    html(render.how_it_works_html())


with tab_eval:
    st.subheader("Model evaluation")
    st.caption(
        "Every number below is read from files written by the evaluation "
        "scripts in scripts/. Nothing here is hard-coded or estimated."
    )

    cross, cross_err = results_view.load_result(
        results_view.CROSS_DATASET_FILE
    )

    st.markdown("#### In-domain vs external dataset")

    if cross is None:
        st.info(cross_err)
    else:
        st.dataframe(
            results_view.comparison_rows(cross),
            hide_index=True,
            use_container_width=True,
        )

        drop = results_view.accuracy_drop_points(cross)
        if drop is not None:
            st.metric("Accuracy drop on WELFake", f"{drop:.1f} points")

        st.caption(
            "The model was trained on ISOT only and evaluated on WELFake "
            "without retraining. WELFake may partly overlap with ISOT-style "
            "sources and has noisy labels, so this can overestimate true "
            "out-of-domain performance."
        )

        confusion = (cross.get("confusion_matrix") or {}).get("matrix")
        if confusion:
            st.markdown("**WELFake confusion matrix** (counts)")
            st.dataframe(
                results_view.confusion_rows(confusion),
                hide_index=True,
                use_container_width=True,
            )

    tuning, tuning_err = results_view.load_result(
        results_view.TUNING_FILE
    )

    st.markdown("#### Threshold tuning (no retraining)")

    if tuning is None:
        st.info(tuning_err)
    else:
        split = tuning.get("split", {})
        st.caption(
            f"Threshold chosen on {split.get('tune_rows', '?')} tuning rows; "
            f"every metric below is computed on the "
            f"{split.get('test_rows', '?')} held-out test rows "
            f"(seed {split.get('seed', '?')})."
        )

        st.dataframe(
            results_view.tuning_rows(tuning),
            hide_index=True,
            use_container_width=True,
        )

        sweep_png = results_view.RESULTS_DIR / "threshold_sweep.png"
        if sweep_png.exists():
            st.image(
                str(sweep_png),
                caption=(
                    "F1 and balanced accuracy vs threshold "
                    "(tuning half only)"
                ),
            )


with tab_about:
    html(render.about_html())
