import sys
import time
import json
import gc
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.asr import get_asr_engine
from app.config import settings

def main():
    audio_path = Path("audio_nlp_1_5min.wav").resolve()
    if not audio_path.exists():
        print(f"Error: {audio_path} does not exist!")
        sys.exit(1)

    eval_dir = Path("eval_results")
    eval_dir.mkdir(exist_ok=True)

    models_to_test = [
        ("cohere", "Cohere Transcribe Arabic (GGUF Q5_K_M)"),
        ("dialectal", "Whisper Large-v3 Turbo Arabic Dialectal (CTranslate2 int8_float16)"),
        ("audar", "Audar ASR V1 Turbo (GGUF Q8_0 + MTMD)"),
    ]

    results = {}

    for model_key, model_label in models_to_test:
        print(f"\n{'='*70}")
        print(f"Starting Evaluation: {model_label} (Key: {model_key})")
        print(f"{'='*70}")

        # Instantiate engine
        engine = get_asr_engine(model_key)

        # 1. Load Model
        t_load_start = time.time()
        print("Loading model weights...")
        engine.load()
        load_time = time.time() - t_load_start
        print(f"Model loaded in {load_time:.2f}s")

        # 2. Transcribe Audio
        t_trans_start = time.time()

        def on_prog(pct, msg):
            print(f"[{model_key.upper()} {pct}%] {msg}", flush=True)

        print("Beginning 5-minute audio transcription...")
        res = engine.transcribe(
            audio_path,
            on_progress=on_prog,
            max_duration_seconds=300.0
        )
        trans_time = time.time() - t_trans_start

        # 3. Metrics
        full_text = res.get("text", "").strip()
        words = res.get("words", [])
        word_count = len(words) or len(full_text.split())
        char_count = len(full_text)
        rtf = trans_time / 300.0  # Real Time Factor

        print(f"\n--- Results for {model_label} ---")
        print(f"Transcription Time: {trans_time:.2f}s (RTF: {rtf:.3f})")
        print(f"Word Count: {word_count}")
        print(f"Character Count: {char_count}")
        print(f"Snippet: {full_text[:200]}...")

        # Save individual transcript
        txt_file = eval_dir / f"transcript_{model_key}.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Saved transcript to {txt_file}")

        results[model_key] = {
            "label": model_label,
            "load_time_sec": round(load_time, 2),
            "transcription_time_sec": round(trans_time, 2),
            "rtf": round(rtf, 4),
            "word_count": word_count,
            "char_count": char_count,
            "words_per_sec": round(word_count / max(1, trans_time), 2),
            "text": full_text,
        }

        # 4. Clean Unload
        print("Unloading engine...")
        engine.unload()
        del engine
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        print(f"Engine {model_key} unloaded. Memory cleared.\n")

    # Save summary json
    summary_file = eval_dir / "asr_comparison_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nAll evaluations complete! Summary saved to {summary_file}")

if __name__ == "__main__":
    main()
