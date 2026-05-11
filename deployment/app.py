"""
ABSA Web Interface (Gradio)
===========================
Run this file to launch a browser-accessible UI for the ABSA model.

In Colab:
    !pip install -q gradio
    %run app.py

Standalone:
    pip install gradio
    python app.py
"""

import json
from inference import predict, ABSAModel

import gradio as gr

# ─── Sentiment styling ────────────────────────────────────────────────────────
SENTIMENT_EMOJI = {
    "positive": "✅",
    "negative": "❌",
    "neutral":  "➖",
    "conflict": "⚡",
}

SENTIMENT_COLOR = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
    "conflict": "#f59e0b",
}

EXAMPLES = [
    "The battery life is outstanding and lasts all day, but the screen is way too dim for outdoor use.",
    "Keyboard feels amazing and the trackpad is super responsive. Build quality is top notch.",
    "Runs hot after 30 minutes of use. The fan is extremely loud and the thermal management is terrible.",
    "Great CPU performance but the RAM is soldered and can't be upgraded. Mixed feelings overall.",
    "Absolutely love this laptop. Fast, lightweight, and the display is gorgeous.",
]


# ─── Core prediction wrapper ──────────────────────────────────────────────────

def run_inference(review_text: str):
    if not review_text.strip():
        return (
            "<p style='color:#94a3b8;font-style:italic;'>Enter a review above.</p>",
            "—",
            "—",
        )

    result = predict(review_text)

    # ── Aspects card HTML ──────────────────────────────────────────────────────
    if not result["valid"] or not result["aspects"]:
        aspects_html = (
            f"<div style='padding:16px;background:#1e293b;border-radius:12px;"
            f"border:1px solid #334155;color:#94a3b8;font-style:italic;'>"
            f"{'No aspects detected.' if result['valid'] else f'Parse error: {result[\"error\"]}'}"
            f"</div>"
        )
    else:
        rows = ""
        for a in result["aspects"]:
            color = SENTIMENT_COLOR.get(a["sentiment"], "#94a3b8")
            emoji = SENTIMENT_EMOJI.get(a["sentiment"], "•")
            rows += (
                f"<div style='display:flex;align-items:center;gap:12px;"
                f"padding:10px 14px;margin-bottom:8px;background:#1e293b;"
                f"border-radius:10px;border-left:3px solid {color};'>"
                f"  <span style='font-size:1.1em;'>{emoji}</span>"
                f"  <span style='flex:1;color:#e2e8f0;font-weight:500;"
                f"font-family:\"DM Mono\",monospace;'>{a['term']}</span>"
                f"  <span style='color:{color};font-size:0.85em;font-weight:600;"
                f"letter-spacing:0.05em;text-transform:uppercase;'>{a['sentiment']}</span>"
                f"</div>"
            )
        count = len(result["aspects"])
        aspects_html = (
            f"<div style='margin-bottom:8px;color:#64748b;font-size:0.8em;"
            f"letter-spacing:0.08em;text-transform:uppercase;'>"
            f"{count} aspect{'s' if count != 1 else ''} detected</div>"
            + rows
        )

    raw_json   = json.dumps({"aspects": result["aspects"]}, indent=2, ensure_ascii=False)
    status     = "✅ Valid" if result["valid"] else f"❌ {result['error']}"

    return aspects_html, raw_json, status


# ─── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui():
    css = """
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

    body, .gradio-container {
        background: #0f172a !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    .title-block {
        text-align: center;
        padding: 40px 0 24px;
    }

    .title-block h1 {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6em;
        color: #f1f5f9;
        letter-spacing: -0.02em;
        margin: 0 0 8px;
    }

    .title-block p {
        color: #64748b;
        font-size: 0.95em;
        font-weight: 300;
        letter-spacing: 0.04em;
    }

    .gr-box, .gr-panel { background: #1e293b !important; border: 1px solid #334155 !important; border-radius: 14px !important; }

    textarea, input[type="text"] {
        background: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.97em !important;
    }

    .gr-button-primary {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.95em !important;
        letter-spacing: 0.03em !important;
        transition: opacity 0.2s !important;
    }

    .gr-button-primary:hover { opacity: 0.88 !important; }

    .gr-button-secondary {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    label { color: #94a3b8 !important; font-size: 0.8em !important; letter-spacing: 0.07em !important; text-transform: uppercase !important; }

    .section-label {
        color: #64748b;
        font-size: 0.75em;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 8px;
        font-weight: 500;
    }

    footer { display: none !important; }
    """

    with gr.Blocks(css=css, title="ABSA — Laptop Review Analyser") as demo:

        gr.HTML("""
            <div class="title-block">
                <h1>Aspect Sentiment Analyser</h1>
                <p>Fine-tuned Qwen 1.5B · LoRA · Laptop domain</p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=5):
                review_input = gr.Textbox(
                    label       = "Laptop Review",
                    placeholder = "e.g. The battery life is great but the keyboard feels cheap...",
                    lines       = 5,
                    max_lines   = 12,
                )
                with gr.Row():
                    submit_btn = gr.Button("Analyse →",  variant="primary", scale=3)
                    clear_btn  = gr.Button("Clear",      variant="secondary", scale=1)

                gr.HTML("<div class='section-label' style='margin-top:16px;'>Examples</div>")
                gr.Examples(
                    examples   = [[e] for e in EXAMPLES],
                    inputs     = [review_input],
                    label      = "",
                )

            with gr.Column(scale=5):
                aspects_output = gr.HTML(
                    label = "Detected Aspects",
                    value = "<p style='color:#334155;font-style:italic;padding:12px;'>Results appear here.</p>",
                )

                with gr.Accordion("Raw JSON output", open=False):
                    json_output = gr.Code(language="json", label="")

                status_output = gr.Textbox(label="Status", interactive=False)

        submit_btn.click(
            fn      = run_inference,
            inputs  = [review_input],
            outputs = [aspects_output, json_output, status_output],
        )
        clear_btn.click(
            fn      = lambda: ("", "<p style='color:#334155;font-style:italic;padding:12px;'>Results appear here.</p>", "", ""),
            inputs  = [],
            outputs = [review_input, aspects_output, json_output, status_output],
        )
        review_input.submit(
            fn      = run_inference,
            inputs  = [review_input],
            outputs = [aspects_output, json_output, status_output],
        )

    return demo


if __name__ == "__main__":
    # Pre-load model before launching UI
    print("⬇️  Loading model (this takes ~30s)...")
    ABSAModel.get()
    print("✅ Model ready. Launching UI...")

    app = build_ui()
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = True,    # creates a public ngrok link — useful in Colab
        show_error  = True,
    )