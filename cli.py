"""Command-line entry point for the OCR pipeline.

Examples:
    python cli.py scan.png                            # local VLM (Ollama), offline
    python cli.py contrat.pdf -o contrat.txt
    python cli.py scan.jpg --engine groq              # cloud VLM instead
    python cli.py scan.jpg --mode hybrid --langs fr ar
    python cli.py scan.png --truth reference.txt      # prints CER/WER
    python cli.py form.pdf --verify --check           # recall pass + format checks
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
    parser.add_argument("--verify", action="store_true",
                        help="second recall pass: re-read the page against the "
                             "draft to recover missed text (~2x slower)")
    parser.add_argument("--check", action="store_true",
                        help="validate postcodes, phone numbers and emails for "
                             "format errors (no ground truth needed)")
    parser.add_argument("--fix", action="store_true",
                        help="apply the unambiguous corrections found by --check "
                             "(implies --check)")
    parser.add_argument("--compare", action="store_true",
                        help="completeness QA: read the page with two local models "
                             "(4b vs 8b) and flag every word they disagree on")
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"file not found: {args.input}")

    if args.compare:
        return _run_compare(args)

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
        result = pipeline.run_pdf(data, language_hint=args.langs, engine=engine,
                                  verify=args.verify)
    else:
        result = pipeline.run_image(data, mode=mode, language_hint=args.langs,
                                    engine=engine, verify=args.verify)

    print(f"[mode={result.mode.value} | engine={result.engine or '-'} "
          f"| langs={','.join(result.languages) or '?'} "
          f"| {result.duration_s}s | preprocessing: {'; '.join(result.preprocessing_steps)}]",
          file=sys.stderr)

    # Corrections are applied before the text is written, so the saved file and
    # any CER/WER measured below reflect what the user actually receives.
    if args.fix:
        from src.validators import apply_fixes
        result.final_text, applied = apply_fixes(result.final_text)
        for f in applied:
            print(f"[fixed] {f.field}: {f.value} -> {f.suggestion}", file=sys.stderr)
        if not applied:
            print("[fixed] nothing unambiguous to correct", file=sys.stderr)

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

    if args.check or args.fix:
        from src.validators import format_report, validate
        print("\n" + format_report(validate(result.final_text)), file=sys.stderr)
    return 0


def _run_compare(args) -> int:
    """Two-model completeness QA: diff 4b vs 8b, flag disagreements to review."""
    from src import compare, local_vlm

    for model in (compare.DEFAULT_MODEL_A, compare.DEFAULT_MODEL_B):
        if not local_vlm.is_available(model):
            parser_error = (f"model '{model}' not available — run: ollama pull {model}")
            print(parser_error, file=sys.stderr)
            return 1

    data = args.input.read_bytes()
    if args.input.suffix.lower() == ".pdf":
        results = compare.compare_pdf(data, languages=args.langs)
    else:
        results = [compare.compare_image(data, languages=args.langs)]

    total_dis = 0
    for i, r in enumerate(results, 1):
        if len(results) > 1:
            print(f"\n===== PAGE {i}/{len(results)} =====")
        print(compare.format_report(r))
        total_dis += len(r.disagreements)

    avg = sum(r.agreement for r in results) / len(results)
    print(f"\n[overall: {len(results)} page(s) | mean agreement {avg*100:.1f}% "
          f"| {total_dis} disagreement span(s) to review]", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
