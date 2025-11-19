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


def build_prompt(system_msg: str, instruction: str, zh_title: str, zh_ingredient: str, candidates: list):
    """
    Build a prompt for ranking candidates based on the given system message, instruction, and query details.

    Args:
        system_msg (str): The system message to set the context.
        instruction (str): The instruction for the ranking task.
        zh_title (str): The Chinese title of the query.
        zh_ingredient (str): The Chinese ingredients of the query.
        candidates (list): A list of candidate dictionaries to be ranked.

    returns:
        str: The constructed prompt.
    """
    prompt = []

    if system_msg:
        prompt.append(f"System: {system_msg}\n")
    prompt.append(f"Instruction: {instruction}\n")
    prompt.append(" ")
    prompt.append(f"Chinese recipe title (Chinese): {zh_title}\n")
    prompt.append(f"Chinese recipe ingredients/description (Chinese): {zh_ingredient}\n")
    prompt.append(" ")
    prompt.append("Options:\n")
    for idx, candidate in enumerate(candidates, start=1):

        #truncate long english description so the propmt isnt huge
        short_en_ingredient = candidate["en_ingredient"][:160]
        prompt.append(f"{idx}. English recipe title: {candidate['en_title']}\n")
    
    prompt.append(" ")
    prompt.append("Answer with the option number (1, 2, 3, ...) of the best matching recipe based on the Chinese title and ingredients/description provided. If none of the options are relevant, answer with 0.\n")

    return "\n".join(prompt)

def call_vllm(base_url: str, api_key: str, model:str, prompt: str, description:dict):
    """
    Call the vLLM API with the given prompt and return the response.

    Args:
        base_url (str): The base URL of the vLLM API. (Optional)
        api_key (str): The API key for authentication. (Optional)
        prompt (str): The prompt to send to the API.
        description (dict): Additional description or metadata.

    Returns:
        str: The response text from the vLLM API.
    """
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": description.get("max_tokens", 8),
        "temperature": description.get("temperature", 0.0),
        "top_p": description.get("top_p", 1.0),
    }

    response = requests.post(f"{base_url}/completions", headers=headers, json=payload, timeout=300)
    response.raise_for_status()
    return resp.json()["choices"][0]["text"].strip()




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

    # build prompt for first query

    system_msg = "You match Chinese recipes to the best English recipe."
    instruction = "You will see a Chinese recipe and several English options. Choose ONE and answer ONLY with its number."
    candidates = groups[first_qid]
    zh_title = candidates[0]["zh_title"]
    zh_ingredient = candidates[0]["zh_ingredient"]
    prompt = build_prompt(system_msg, instruction, zh_title, zh_ingredient, candidates[:5])
    
    print("==== PROMPT START ====")
    print(prompt)
    print("==== PROMPT END ====")

    # call vLLM API 
    base_url = "http://localhost:8000/v1"  # Replace with your vLLM API URL
    api_key = ""  # Replace with your API key if needed
    model = "Qwen/Qwen2.5-1.5B-Instruct"  # Replace with your model name
    prompt = prompt
    description = {"max_tokens": 8, "temperature": 0.0, "top_p": 1.0}

    out = call_vllm(base_url, api_key, model, prompt, description)
    print("vLLM response:", out)



