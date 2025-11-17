import argparse
import json
from pathlib import Path
import multiprocessing as mp
from functools import partial

from tqdm import tqdm

from src.utils.datasets import CollectionParser
from src.utils.defaults import COLLECTION_TYPES


def process_line(line, collection_type, num_special_tokens):
    doc_id, doc = CollectionParser.parse(line, collection_type)
    doc += " " + " ".join(f"<EXP{i}>" for i in range(num_special_tokens))
    return f"{doc_id}\t{doc}\n"


def merge_collection_and_expansions(collection_path: Path, collection_type: str, num_special_tokens: int, output: Path):
    with open(collection_path) as f:
        lines = f.readlines()
    # print(lines)
        
    with mp.Pool() as pool:
        process_func = partial(process_line, collection_type=collection_type, num_special_tokens=num_special_tokens)
        results = list(tqdm(pool.imap(process_func, lines), total=len(lines)))
    
    with open(output, 'w') as out:
        for result in results:
            out.write(result)



def merge_collection_and_expansions_single_process(collection_path: Path, collection_type: str, num_special_tokens: int, output: Path):
    new_docs = []
    with open(collection_path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            doc_id, doc = CollectionParser.parse(line, collection_type)
            doc += " " + " ".join(f"<EXP{i}>" for i in range(num_special_tokens))
            new_docs.append(f"{doc_id}\t{doc}\n")
    with open(output, 'w+') as out:
        for doc in new_docs:
            out.write(doc)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collection with generated queries')
    parser.add_argument('--collection_path', type=Path)
    parser.add_argument('--collection_type', type=str, choices=COLLECTION_TYPES)
    parser.add_argument('--num_special_tokens', type=int)
    parser.add_argument('--output_path', type=Path)
    args = parser.parse_args()

    merge_collection_and_expansions_single_process(args.collection_path, args.collection_type, args.num_special_tokens, args.output_path)
