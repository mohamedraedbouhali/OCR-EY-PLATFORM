"""Command-line entry point for the OCR pipeline.

Examples:
    python cli.py scan.png                            # local VLM (Ollama), offline
    python cli.py contrat.pdf -o contrat.txt
    python cli.py scan.jpg --engine groq              # cloud VLM instead
    python cli.py scan.jpg --mode hybrid --langs fr ar
    python cli.py scan.png --truth reference.txt      # prints CER/WER
"""

import argparse
import sys
from pathlib import Path

from src.config import DEFAULT_ENGINE, OLLAMA_MODEL, SUPPORTED_LANGUAGES, api_key_present
from src.evaluation import accuracy_summary
from src.pipeline import Engine, Mode, OCRPipeline


def main() -> int:
    # Windows consoles default to a legacy code page (cp1252) that cannot
    # represent Arabic or some accented output — force UTF-8 on both streams.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Intelligent multilingual OCR (FR/EN/AR)")
    parser.add_argument("input", type=Path, help="scanned image (png/jpg/webp) or PDF")
    parser.add_argument("-o", "--output", type=Path, help="write text to this file")
    parser.add_argument("--mode", choices=[m.value for m in Mode],
                        default=Mode.GENAI_VISION.value)
    parser.add_argument("--engine", choices=[e.value for e in Engine],
                        default=DEFAULT_ENGINE,
                        help="VLM backend: local (Ollama, offline) or groq (cloud)")
    parser.add_argument("--langs", nargs="*", choices=list(SUPPORTED_LANGUAGES),
                        help="language hint, e.g. --langs fr ar")
    parser.add_argument("--truth", type=Path,
                        help="ground-truth text file for CER/WER evaluation")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"file not found: {args.input}")
    mode = Mode(args.mode)
    engine = Engine(args.engine)
    if mode != Mode.TESSERACT_ONLY:
        if engine is Engine.GROQ and not api_key_present():
            parser.error("GROQ_API_KEY is not set (see .env.example)")
        if engine is Engine.LOCAL:
            from src import local_vlm
            if not local_vlm.is_available():
                parser.error(f"local VLM unavailable — start Ollama and run: "
                             f"ollama pull {OLLAMA_MODEL}")

    pipeline = OCRPipeline()
    data = args.input.read_bytes()
    if args.input.suffix.lower() == ".pdf":
        result = pipeline.run_pdf(data, language_hint=args.langs, engine=engine)
    else:
        result = pipeline.run_image(data, mode=mode, language_hint=args.langs,
                                    engine=engine)

    print(f"[mode={result.mode.value} | engine={result.engine or '-'} "
          f"| langs={','.join(result.languages) or '?'} "
          f"| {result.duration_s}s | preprocessing: {'; '.join(result.preprocessing_steps)}]",
          file=sys.stderr)

    if args.output:
        args.output.write_text(result.final_text, encoding="utf-8")
        print(f"Saved -> {args.output}", file=sys.stderr)
    else:
        print(result.final_text)

    if args.truth:
        s = accuracy_summary(args.truth.read_text(encoding="utf-8"), result.final_text)
        print(f"CER={s['cer']:.2%}  WER={s['wer']:.2%}  "
              f"char_acc={s['char_accuracy']}%  word_acc={s['word_accuracy']}%",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
