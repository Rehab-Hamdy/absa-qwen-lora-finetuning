"""
ABSA Inference Pipeline
=======================
Loads the fine-tuned Qwen LoRA model and runs aspect-based sentiment analysis.

Usage:
    python inference.py                          # starts FastAPI server
    python inference.py --text "Your review"     # single inference from CLI
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Paths — edit these to match your Drive layout ───────────────────────────
BASE_MODEL_PATH = "/content/drive/MyDrive/NLU_Finetuning/model"
ADAPTER_PATH    = "/content/drive/MyDrive/NLU_Finetuning/model/qwen_absa_lora"

SENTIMENT_CLASSES = ["positive", "negative", "neutral", "conflict"]

SYSTEM_PROMPT = """You are an expert Aspect-Based Sentiment Analysis (ABSA) system.
Your ONLY job is to extract aspect terms from laptop reviews and assign a sentiment to each.

RULES:
1. Return ONLY a valid JSON object — no explanation, no markdown, no extra text.
2. The JSON must follow this exact schema:
   {"aspects": [{"term": "<aspect>", "sentiment": "<sentiment>"}]}
3. Valid sentiment values: positive | negative | neutral | conflict
4. "conflict" means the review expresses both positive and negative opinions about the same aspect.
5. Extract ALL mentioned aspects, even if the sentiment is neutral.
6. If no aspects are found, return: {"aspects": []}
7. Use lowercase for both term and sentiment.""".strip()

FEW_SHOT_EXAMPLES = [
    {
        "review": "The battery life is amazing and lasts all day, but the screen is quite dim.",
        "output": json.dumps({
            "aspects": [
                {"term": "battery life", "sentiment": "positive"},
                {"term": "screen",       "sentiment": "negative"},
            ]
        }, separators=(",", ":")),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Model loader (singleton — loads once, reused for every request)
# ─────────────────────────────────────────────────────────────────────────────

class ABSAModel:
    _instance: Optional["ABSAModel"] = None

    def __init__(self):
        self.tokenizer = None
        self.model     = None
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.gen_config = None

    @classmethod
    def get(cls) -> "ABSAModel":
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load()
        return cls._instance

    def _load(self):
        log.info(f"Loading tokenizer from {ADAPTER_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_PATH,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token    = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        log.info(f"Loading base model from {BASE_MODEL_PATH} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype       = torch.float32,
            trust_remote_code = True,
            attn_implementation = "eager",
        ).to(self.device)

        log.info(f"Loading LoRA adapter from {ADAPTER_PATH} ...")
        self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        self.model.eval()

        self.gen_config = GenerationConfig(
            max_new_tokens      = 128,
            do_sample           = False,
            temperature         = 1.0,
            repetition_penalty  = 1.1,
            pad_token_id        = self.tokenizer.pad_token_id,
            eos_token_id        = self.tokenizer.eos_token_id,
        )

        log.info(f"✅ Model ready on {self.device}  "
                 f"({sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params)")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_prompt(review_text: str, tokenizer) -> str:
    few_shot_block = "\n".join(
        f"Review: {ex['review']}\nOutput: {ex['output']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    user_content = (
        f"Examples:\n{few_shot_block}\n\n"
        f"Now analyze the following review:\n"
        f"Review: {review_text}\n"
        f"Output:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_output(raw: str) -> Dict[str, Any]:
    """
    Parse raw model output into structured aspects.
    Returns {"aspects": [...], "raw": str, "valid": bool, "error": str|None}
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)

    start = raw.find("{")
    if start == -1:
        return {"aspects": [], "raw": raw, "valid": False, "error": "No JSON found in output"}

    brace_count = 0
    end = start
    for i in range(start, len(raw)):
        if raw[i] == "{": brace_count += 1
        elif raw[i] == "}": brace_count -= 1
        if brace_count == 0:
            end = i + 1
            break

    candidate = raw[start:end]
    candidate = re.sub(r",\s*}", "}", candidate)
    candidate = re.sub(r",\s*]", "]", candidate)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        return {"aspects": [], "raw": raw, "valid": False, "error": f"JSON parse error: {e}"}

    if "aspects" not in data or not isinstance(data["aspects"], list):
        return {"aspects": [], "raw": raw, "valid": False, "error": "Missing 'aspects' key"}

    cleaned = []
    for item in data["aspects"]:
        term      = str(item.get("term", "")).strip().lower()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if not term:
            continue
        if sentiment not in SENTIMENT_CLASSES:
            sentiment = "neutral"
        cleaned.append({"term": term, "sentiment": sentiment})

    return {"aspects": cleaned, "raw": raw, "valid": True, "error": None}


# ─────────────────────────────────────────────────────────────────────────────
# Core inference function
# ─────────────────────────────────────────────────────────────────────────────

def predict(review_text: str) -> Dict[str, Any]:
    """
    Run ABSA inference on a single review string.

    Returns:
        {
            "aspects": [{"term": str, "sentiment": str}, ...],
            "raw_output": str,
            "valid": bool,
            "error": str | None,
        }
    """
    absa = ABSAModel.get()

    prompt = build_inference_prompt(review_text, absa.tokenizer)
    inputs = absa.tokenizer(
        prompt,
        return_tensors  = "pt",
        truncation      = True,
        max_length      = 256,
        padding         = False,
    ).to(absa.device)

    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = absa.model.generate(
            **inputs,
            generation_config = absa.gen_config,
        )

    # Decode only the newly generated tokens (strip the prompt)
    new_ids    = output_ids[0][prompt_len:]
    raw_output = absa.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    result = parse_output(raw_output)
    return {
        "aspects"    : result["aspects"],
        "raw_output" : raw_output,
        "valid"      : result["valid"],
        "error"      : result["error"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI web server
# ─────────────────────────────────────────────────────────────────────────────

def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel as PydanticBase
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    app = FastAPI(
        title       = "ABSA Inference API",
        description = "Aspect-Based Sentiment Analysis using fine-tuned Qwen + LoRA",
        version     = "1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins  = ["*"],
        allow_methods  = ["*"],
        allow_headers  = ["*"],
    )

    class ReviewRequest(PydanticBase):
        text: str

    class AspectResult(PydanticBase):
        term      : str
        sentiment : str

    class PredictResponse(PydanticBase):
        aspects    : List[AspectResult]
        raw_output : str
        valid      : bool
        error      : Optional[str]

    @app.on_event("startup")
    def load_model():
        ABSAModel.get()   # load once at server start

    @app.get("/health")
    def health():
        return {"status": "ok", "device": ABSAModel.get().device}

    @app.post("/predict", response_model=PredictResponse)
    def predict_endpoint(req: ReviewRequest):
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        result = predict(req.text)
        return PredictResponse(**result)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA Inference")
    parser.add_argument("--text",   type=str, default=None, help="Single review to analyse")
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    args = parser.parse_args()

    if args.text:
        # CLI mode — single inference
        print("\n🔍 Analysing review...")
        result = predict(args.text)
        print(f"\nReview    : {args.text}")
        print(f"Raw output: {result['raw_output']}")
        print(f"Valid     : {result['valid']}")
        if result["error"]:
            print(f"Error     : {result['error']}")
        print("\nAspects:")
        if result["aspects"]:
            for a in result["aspects"]:
                emoji = {"positive": "✅", "negative": "❌", "neutral": "➖", "conflict": "⚡"}.get(a["sentiment"], "•")
                print(f"  {emoji}  {a['term']:30s}  →  {a['sentiment']}")
        else:
            print("  (no aspects detected)")
    else:
        # Server mode
        import uvicorn
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)