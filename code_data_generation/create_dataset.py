from dotenv import load_dotenv
from chunking_evaluation.evaluation_framework.synthetic_evaluation import SyntheticEvaluation
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(dotenv_path=BASE_DIR.parent / ".env")
API_KEY = os.environ.get('OPENAI_API_KEY')

FILENAMES = [
    "attrs.md",
    "click.md",
    "pendulum.md",
    "marshmallow.md",
    "tabulate.md"
]

corpora_paths = [str(BASE_DIR / "corpora_with_headers" /  filename) for filename in FILENAMES]
queries_csv_path = BASE_DIR / "generated_queries_excerpts.csv"

def main():
    evaluation = SyntheticEvaluation(
        corpora_paths,
        str(queries_csv_path),
        openai_api_key=API_KEY
    )

    evaluation.generate_queries_and_excerpts(num_rounds=2, queries_per_corpus=2) 
    print("\nFinished generating initial queries and excerpts\n")

    print("Query-excerpt dataset generated!")

if __name__ == "__main__":
    main() 