#!/usr/bin/env python3
import json
import re
import unicodedata
import warnings
import pandas as pd
from typing import List, Tuple
from pathlib import Path

# Since Headers were used during data creation in order to help the LLM better understand which files it was working with, 
# the originally generated dataset's reference content when generated is misaligned with its respective start/end indicies in the corpus

class AlignmentError(Exception):
    pass

def normalize_ref_text(raw: str) -> str:
    """
    Decode JSON-style escapes into real newlines/quotes,
    then normalize Unicode to NFC form.
    """
    decoded = raw.encode('utf-8').decode('unicode_escape')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
    return unicodedata.normalize('NFC', decoded)

def find_reference_ranges(
    corpus: str,
    references: List[dict]
) -> List[Tuple[int, int]]:
    """
    For each reference dict with a 'content' field,
    locate its exact slice in `corpus` (advancing a cursor),
    falling back to whitespace-insensitive regex if needed.
    """
    cursor = 0
    spans: List[Tuple[int, int]] = []

    for ref in references:
        text = normalize_ref_text(ref['content'])
        # 1) exact forward search
        idx = corpus.find(text, cursor)
        if idx != -1:
            start, end = idx, idx + len(text)
        else:
            # 2) whitespace-insensitive fallback
            parts = [re.escape(piece) for piece in text.split()]
            pattern = re.compile(r'\s+'.join(parts), flags=re.MULTILINE)
            m = pattern.search(corpus, cursor)
            if not m:
                raise AlignmentError(
                    f"Failed to align snippet starting at cursor {cursor}: {text[:30]!r}..."
                )
            start, end = m.span()
        spans.append((start, end))
        cursor = end

    return spans

def main():
    base_dir = Path(__file__).resolve().parent
    corp_dir = base_dir / "evaluation_framework" / "code_evaluation_data" / "corpora"
    csv_file = base_dir / "evaluation_framework" / "code_evaluation_data" / "code_questions_df.csv"
    df = pd.read_csv(csv_file, dtype=str)

    cache = {}

    for idx, row in df.iterrows():
        cid = row["corpus_id"]
        # 1) Load or fetch cached corpus_text
        if cid not in cache:
            p = corp_dir / f"{cid}.md"
            cache[cid] = unicodedata.normalize("NFC", p.read_text(encoding="utf-8"))
        corpus_text = cache[cid]

        # 2) Parse JSON & align
        refs = json.loads(row["references"])
        try:
            spans = find_reference_ranges(corpus_text, refs)
        except AlignmentError as e:
            csv_row = idx + 2  
            print(f"[CSV Row {csv_row}, corpus={cid}] Alignment failed: {e}")          
            continue

        # 3) Inject new spans & dump JSON back into DataFrame
        for ref_obj, (s,e) in zip(refs, spans):
            ref_obj["start_index"], ref_obj["end_index"] = s, e
        df.at[idx, "references"] = json.dumps(refs, ensure_ascii=False)

    # 6) Save to a new CSV
    out_path = base_dir / "evaluation_framework" / "code_evaluation_data" / "code_questions_df_aligned.csv"
    df.to_csv(out_path, index=False, encoding='utf-8')
    print(f"Alignment complete. Aligned CSV written to:\n  {out_path}")

if __name__ == "__main__":
    main()