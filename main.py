import os
import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from chunking_evaluation import SyntheticEvaluation, BaseChunker

from chunking_evaluation.chunking import (
    ClusterSemanticChunker,
    FixedTokenChunker,
    KamradtModifiedChunker,
    LLMSemanticChunker,
    RecursiveTokenChunker,
)

from langchain.text_splitter import PythonCodeTextSplitter, TokenTextSplitter

from chromadb.utils import embedding_functions

this_file = Path(__file__).resolve()
BASE_DIR = this_file.parent
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")

QUERIES_CSV_PATH = (
    BASE_DIR
    / "chunking_evaluation"
    / "evaluation_framework"
    / "code_evaluation_data"
    / "code_questions_df.csv"
)
evaluation_data_dir = (
    BASE_DIR
    / "chunking_evaluation"
    / "evaluation_framework"
    / "code_evaluation_data"
    / "corpora"
) 

corpora_paths = list(evaluation_data_dir.glob("*.md"))
corpora_id_paths = {path.stem: path for path in corpora_paths}
DB_PATH = BASE_DIR / "evaluation_results.db"

def create_tables(cursor):
    # Detailed per-question scores
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS detailed_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            chunker TEXT,
            embedding TEXT,
            chunk_size INTEGER,
            chunk_overlap INTEGER,
            corpus_id TEXT,
            question_idx INTEGER,
            precision_omega REAL,
            iou REAL,
            recall REAL,
            precision REAL
        );
        """
    )
    # Summary statistics for the run and mean scores
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS summary_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            chunker TEXT,
            embedding TEXT,
            chunk_size INTEGER,
            chunk_overlap INTEGER,
            precision_omega_mean REAL,
            precision_omega_std REAL,
            iou_mean REAL,
            iou_std REAL,
            recall_mean REAL,
            recall_std REAL,
            precision_mean REAL,
            precision_std REAL
        );
        """
    )

def main():
    # Initialize chunker and evaluation framework
    API_KEY = os.getenv("OPENAI_API_KEY")

    # Create a List[str] from list[Path]
    corpora_path_strings = []
    for key in corpora_id_paths.keys():
        corpora_path_strings.append(key)

    # Set chunker
    chunker = PythonCodeTextSplitter(chunk_size=800, chunk_overlap=0)

    evaluation = SyntheticEvaluation(
        corpora_paths=corpora_path_strings,  # originally passed as list[Path] ("corpora_paths")
        queries_csv_path=str(QUERIES_CSV_PATH),  # originally passed as Path
        openai_api_key=API_KEY,
        corpora_id_paths=corpora_id_paths,
    )

    default_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=API_KEY,
        model_name="text-embedding-3-large"
    )

    results = evaluation.run(chunker, default_ef)

    # Metadata for this run
    timestamp = datetime.now(ZoneInfo("Europe/Stockholm")).isoformat()    
    chunker_name = chunker.__class__.__name__
    embedding_name = default_ef.__class__.__name__
    chunk_size = getattr(chunker, '_chunk_size', None)
    chunk_overlap = getattr(chunker, '_chunk_overlap', None)

    # Connect to database and ensure tables exist
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    create_tables(cursor)

    # Insert detailed per-question and per-corpus scores
    corpora_scores = results.get('corpora_scores', {})
    for corpus_id, scores in corpora_scores.items():
        n = len(scores.get('iou_scores', []))
        for idx in range(n):
            cursor.execute(
                """
                INSERT INTO detailed_results (
                    timestamp, chunker, embedding, chunk_size, chunk_overlap,
                    corpus_id, question_idx,
                    precision_omega, iou, recall, precision
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, chunker_name, embedding_name, chunk_size, chunk_overlap,
                    corpus_id, idx,
                    scores['precision_omega_scores'][idx],
                    scores['iou_scores'][idx],
                    scores['recall_scores'][idx],
                    scores['precision_scores'][idx],
                ),
            )

    cursor.execute(
        """
        INSERT INTO summary_results (
            timestamp, chunker, embedding, chunk_size, chunk_overlap,
            precision_omega_mean, precision_omega_std,
            iou_mean, iou_std,
            recall_mean, recall_std,
            precision_mean, precision_std
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp, chunker_name, embedding_name, chunk_size, chunk_overlap,
            results.get('precision_omega_mean'), results.get('precision_omega_std'),
            results.get('iou_mean'), results.get('iou_std'),
            results.get('recall_mean'), results.get('recall_std'),
            results.get('precision_mean'), results.get('precision_std'),
        ),
    )

    conn.commit()
    conn.close()

    print(f"Saved detailed and summary results to {DB_PATH}")

if __name__ == "__main__":
    main()
