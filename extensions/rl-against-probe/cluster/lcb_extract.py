"""Download LiveCodeBench code_generation_lite (all releases) and write medium+hard problems with their tests,
in the shape ImpossibleBench's transcriber consumes (statement, test cases), to lcb_medhard.jsonl. ImpossibleBench
dropped any problem whose full test set exceeded 40k chars; we instead drop oversized individual tests."""
import base64, json, pickle, zlib
from huggingface_hub import hf_hub_download, list_repo_files
REPO = "livecodebench/code_generation_lite"
files = sorted(f for f in list_repo_files(REPO, repo_type="dataset") if f.endswith(".jsonl"))
print(files, flush=True)
seen, out = set(), open("/data/agastyas/rh_rl/lcb_medhard.jsonl", "w")
n = {"medium": 0, "hard": 0}
for f in files:
    p = hf_hub_download(REPO, f, repo_type="dataset")
    for l in open(p):
        u = json.loads(l)
        if u["difficulty"] not in n or u["question_id"] in seen:
            continue
        seen.add(u["question_id"])
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(u["private_test_cases"]))))
        pub = json.loads(u["public_test_cases"])
        tests, size = [], 0
        for w in pub + priv:  # keep tests that fit: each <= 3k chars, total < 40k (the transcriber's prompt budget)
            s = len(w["input"]) + len(w["output"])
            if s <= 3000 and size + s < 40000:
                tests.append(w); size += s
        if len(tests) < 5:
            continue
        out.write(json.dumps({"question_id": u["question_id"], "difficulty": u["difficulty"], "platform": u["platform"],
                              "contest_date": u["contest_date"], "title": u["question_title"],
                              "statement": u["question_content"], "starter_code": u["starter_code"], "tests": tests}) + "\n")
        n[u["difficulty"]] += 1
    print(f, n, flush=True)
print("DONE", n)
