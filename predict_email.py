import argparse
import sys
from pathlib import Path
import joblib
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def basic_clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"[0-9]+", " NUMBER ", text)
    text = re.sub(r"[\w\.-]+@[\w\.-]+", " EMAIL ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

_model_cache = None

def load_model(model_path: Path):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not model_path.exists():
        logger.error(f"Nie znaleziono modelu: {model_path}")
        sys.exit(1)
    _model_cache = joblib.load(model_path)
    return _model_cache

def predict_text(model, text: str, threshold: float):
    clean = basic_clean(text)
    proba = model.predict_proba([clean])[0][1]
    pred = int(proba >= threshold)
    return pred, proba

def predict_from_file(model, file_path: Path, threshold: float):
    if not file_path.exists():
        logger.error(f"Nie znaleziono pliku: {file_path}")
        sys.exit(1)
    with file_path.open(encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            pred, proba = predict_text(model, line, threshold)
            label = "Phishing" if pred == 1 else "Safe"
            print(f"{idx:>3}: {label:<8} | {proba:.1%} | {line[:50]}{'...' if len(line)>50 else ''}")

def main():
    parser = argparse.ArgumentParser(
        description="Klasyfikacja maili jako bezpieczne lub phishingowe."
    )
    parser.add_argument(
        '-m', '--model', type=Path,
        default=Path(__file__).parent / 'outputs' / 'model_LR_advanced.joblib',
        help="Ścieżka do wytrenowanego modelu (joblib)."
    )
    parser.add_argument(
        '-t', '--text', type=str,
        help="Pojedyncza treść maila do analizy."
    )
    parser.add_argument(
        '-f', '--file', type=Path,
        help="Plik tekstowy z mailami (po jednej wiadomości na linii)."
    )
    parser.add_argument(
        '-T', '--threshold', type=float, default=0.5,
        help="Próg prawdopodobieństwa do uznania za phishing (domyślnie 0.5)."
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.file:
        predict_from_file(model, args.file, args.threshold)
        return

    if args.text:
        mail = args.text
    else:
        logger.info("Wklej treść maila, zakończ CTRL+D:")
        mail = sys.stdin.read()
    if not mail.strip():
        logger.error("Brak tekstu do analizy.")
        sys.exit(1)

    pred, proba = predict_text(model, mail, args.threshold)
    label = "Phishing Email" if pred == 1 else "Safe Email"
    print(f"Predykcja: {label}")
    print(f"Prawdopodobieństwo phishingu: {proba:.2%}")

if __name__ == '__main__':
    main()
