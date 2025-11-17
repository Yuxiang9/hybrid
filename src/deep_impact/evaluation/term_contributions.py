from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

from tqdm.auto import tqdm

from src.deep_impact.inverted_index.inverted_index import InvertedIndex
from src.deep_impact.models import DeepImpact
from src.utils.datasets import Queries, QueryRelevanceDataset, RunFile


def get_query_terms(query: str):
    """Tokenize query using DeepImpact model preprocessing."""
    # DeepImpact provides a static `process_query` method returning a set of terms
    return DeepImpact.process_query(query)


def load_first_ranked_docs(run_file_path: Union[str, Path]) -> Dict[int, Tuple[int, float]]:
    """Return mapping qid -> (pid, score) for the first-ranked doc per query."""
    run_reader = RunFile(run_file_path)
    first_docs: Dict[int, Tuple[int, float]] = {}
    for qid, pid, rank, score in run_reader.read():
        # keep the first encountered rank==1 or minimal rank
        if qid not in first_docs or rank < first_docs[qid][2]:  # store rank to compare
            first_docs[qid] = (pid, score, rank)
    # remove rank from tuple
    return {qid: (pid, score) for qid, (pid, score, _) in first_docs.items()}


def term_scores_for_doc(index: InvertedIndex, doc_id: int, query_terms: set) -> List[Tuple[str, float]]:
    """Return list of (term, score) for terms from query_terms contributing to doc_id."""
    contributions = []
    for term in query_terms:
        for d_id, value in index.term_docs(term):
            if d_id == doc_id:
                contributions.append((term, value))
                break
    # sort descending by score
    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions


def analyze_term_contributions(
        index_path: Union[str, Path],
        queries_path: Union[str, Path],
        qrels_path: Union[str, Path],
        run_file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
):
    index = InvertedIndex(index_path)
    queries = Queries(queries_path)
    qrels = QueryRelevanceDataset(qrels_path)

    first_docs = load_first_ranked_docs(run_file_path)

    out_lines = []
    count = 0
    for qid, (pid, score) in tqdm(first_docs.items(), desc="Analyzing term contributions"):
        # consider only correctly ranked positives
        if pid not in qrels[qid]:
            continue
        query_terms = get_query_terms(queries[qid])
        contributions = term_scores_for_doc(index, pid, query_terms)
        # Format: qid pid term score
        for term, tscore in contributions:
            out_lines.append(f"{qid}\t{pid}\t{term}\t{tscore}\n")
        count += 1
        if count > 100:
            break

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("qid\tpid\tterm\tterm_score\n")
            f.writelines(out_lines)
    else:
        print("qid\tpid\tterm\tterm_score")
        for line in out_lines:
            print(line.strip())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze which terms contributed to correctly ranked positive documents.")
    parser.add_argument("--index_path", required=True, help="Path to inverted index directory")
    parser.add_argument("--queries_path", required=True, help="Path to queries tsv file")
    parser.add_argument("--qrels_path", required=True, help="Path to qrels file")
    parser.add_argument("--run_file_path", required=True, help="Path to run file")
    parser.add_argument("--output_path", "-o", help="Optional output TSV path")

    args = parser.parse_args()

    analyze_term_contributions(
        args.index_path,
        args.queries_path,
        args.qrels_path,
        args.run_file_path,
        args.output_path,
    )
