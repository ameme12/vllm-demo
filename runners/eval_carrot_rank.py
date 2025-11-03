# runners/eval_carrot_rank.py

import csv
import json
import sys
import yaml
import pathlib
import requests
from collections import defaultdict

def load_carrot_group(tsv_path: str):
    """
    Read IR_Dataset/test_set.tsv and group rows by query_id (col 0)

    Returns:
        dict: A dictionary where keys are query_ids and values are lists of rows.

    """

    groups = defaultdict(list)
    with open(tsv_path, newline="") as f:
        reader = csv.reader(f, delimiter = "\t")
        for row in reader:
            if not row:
                continue
            query_id = row[0]
            relevance = int(row[1])
            candidate_id = row[2] if len(row)> 2 else ""
            tag= row[3] if len(row)>3 else ""
            zh_title = row[4] if len(row)>4 else ""
            zh_ingredient = row[5] if len(row)>5 else ""
            en_title = row[6] if len(row)>6 else ""
            en_ingredient = row[7] if len(row)>7 else ""

            #print(query_id, relevance)

            groups[query_id].append(
                {
                    "relevance": relevance,
                    "candidate_id": candidate_id,
                    "tag": tag,
                    "zh_title": zh_title,
                    "zh_ingredient": zh_ingredient,
                    "en_title": en_title,
                    "en_ingredient": en_ingredient,
                }
            )

    return groups



if __name__ == "__main__":
    # quick manual test
    groups = load_carrot_group("data/CARROT-Task/IR_Dataset/test_set.tsv")
    print("num queries:", len(groups))

    # print first query to see structure
    first_qid = next(iter(groups))
    print("first query id:", first_qid)
    print("first query candidates:")
    for c in groups[first_qid][:3]:
        print(c)


