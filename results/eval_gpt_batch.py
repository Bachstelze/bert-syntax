from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizer
import torch
import sys
import csv
import logging
import itertools
import numpy as np
from collections import Counter, defaultdict
import time

model_name = "gpt2"
print(f"using model: {model_name}", file=sys.stderr)
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

split_words = 'no_split' not in sys.argv
use_postfix = 'use_postfix' in sys.argv
reduce_tokens = 'reduce_tokens' in sys.argv

# Use eos_token_id as the padding token
padding_token_id = tokenizer.eos_token_id

def get_probs_for_words_batch(sentences, w1_list, w2_list):
    input_ids_batch = []
    w1_ids_batch = []
    w2_ids_batch = []

    for sent, w1, w2 in zip(sentences, w1_list, w2_list):
        pre, target, post = sent.split("***")
        target = ["[MASK]"] if "mask" in target.lower() else tokenizer.tokenize(target)

        tokens = tokenizer.tokenize(pre)
        target_idx = len(tokens)

        if reduce_tokens:
          # we reduce the GPT vocab to the the tokens which are in the BERT tokenizer
          try:
              word_ids = bert_tokenizer.convert_tokens_to_ids([w1, w2])
          except KeyError:
              print("skipping", w1, w2, "bad tokens")
              continue

        tok_w1, tok_w2 = tokenizer.tokenize(w1), tokenizer.tokenize(w2)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        w1_ids = tokenizer.convert_tokens_to_ids(tok_w1)
        w2_ids = tokenizer.convert_tokens_to_ids(tok_w2)

        if not input_ids:
            print("skipping", pre, w1, w2, "empty beginning")
            continue

        if not split_words and (len(tok_w1) > 1 or len(tok_w2) > 1):
            print("skipping", pre, w1, w2, "splitted words")
            continue

        if use_postfix:
            end_tokens = tokenizer.tokenize(post)
            end_ids = tokenizer.convert_tokens_to_ids(end_tokens)
            w1_ids += end_ids
            w2_ids += end_ids

        input_ids_batch.append(input_ids)
        w1_ids_batch.append(w1_ids)
        w2_ids_batch.append(w2_ids)

    if not input_ids_batch:
        return None

    # Pad the sequences to the max length in the batch
    max_len = max(len(ids) for ids in input_ids_batch)
    padded_input_ids_batch = []
    padded_w1_ids_batch = []
    padded_w2_ids_batch = []

    for input_ids, w1_ids, w2_ids in zip(input_ids_batch, w1_ids_batch, w2_ids_batch):
        padded_input_ids = input_ids + [padding_token_id] * (max_len - len(input_ids))
        padded_w1_ids = w1_ids + [padding_token_id] * (max_len - len(w1_ids))
        padded_w2_ids = w2_ids + [padding_token_id] * (max_len - len(w2_ids))

        padded_input_ids_batch.append(padded_input_ids)
        padded_w1_ids_batch.append(padded_w1_ids)
        padded_w2_ids_batch.append(padded_w2_ids)

    # Convert lists to tensors
    tens_input = torch.LongTensor(padded_input_ids_batch).to(device)
    tens_w1 = torch.LongTensor(padded_w1_ids_batch).to(device)
    tens_w2 = torch.LongTensor(padded_w2_ids_batch).to(device)

    with torch.no_grad():
        res = model(tens_input)[0]

        res = res[..., 0:model.config.vocab_size]
        res = torch.nn.functional.log_softmax(res, dim=-1)

    # Use masking to avoid including padding tokens in score calculation
    mask_w1 = (tens_w1 != padding_token_id)
    mask_w2 = (tens_w2 != padding_token_id)

    scores_w1 = res.gather(2, tens_w1.unsqueeze(-1)).squeeze(-1)
    scores_w2 = res.gather(2, tens_w2.unsqueeze(-1)).squeeze(-1)

    masked_scores_w1 = scores_w1 * mask_w1.float()
    masked_scores_w2 = scores_w2 * mask_w2.float()

    total_scores_w1 = masked_scores_w1.sum(dim=1).cpu().numpy()
    total_scores_w2 = masked_scores_w2.sum(dim=1).cpu().numpy()

    return total_scores_w1, total_scores_w2

def load_marvin():
    cc = Counter()
    out = []
    for line in open("marvin_linzen_dataset.tsv"):
        case = line.strip().split("\t")
        cc[case[1]] += 1
        g, ug = case[-2], case[-1]
        g, ug = g.split(), ug.split()
        assert len(g) == len(ug), (g, ug)
        diffs = [i for i, pair in enumerate(zip(g, ug)) if pair[0] != pair[1]]
        if len(diffs) != 1:
            continue
        gv, ugv = g[diffs[0]], ug[diffs[0]]
        g[diffs[0]] = "***mask***"
        g.append(".")
        out.append((case[0], case[1], " ".join(g), gv, ugv))
    return out

def eval_marvin():
    o = load_marvin()
    print(len(o), file=sys.stderr)
    start = time.time()

    cases, types, sentences, goods, bads = zip(*o)
    batch_size = 16  # Adjust this based on your hardware

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_goods = goods[i:i+batch_size]
        batch_bads = bads[i:i+batch_size]

        ps = get_probs_for_words_batch(batch_sentences, batch_goods, batch_bads)
        if ps is None:
            continue
        gp_batch, bp_batch = ps
        for j in range(len(gp_batch)):
            print(gp_batch[j] > bp_batch[j], cases[i+j], types[i+j], batch_goods[j], batch_bads[j], batch_sentences[j])

        if i % 100 == 0:
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()

def eval_lgd():
    lines = [line.strip().split("\t") for line in open("lgd_dataset.tsv", encoding="utf8")]
    na_list, masked_list, good_list, bad_list = zip(*[(line[0], line[2], line[3], line[4]) for line in lines])
    batch_size = 16  # Adjust this based on your hardware

    for i in range(0, len(masked_list), batch_size):
        batch_masked = masked_list[i:i+batch_size]
        batch_goods = good_list[i:i+batch_size]
        batch_bads = bad_list[i:i+batch_size]

        ps = get_probs_for_words_batch(batch_masked, batch_goods, batch_bads)
        if ps is None:
            continue
        gp_batch, bp_batch = ps
        for j in range(len(gp_batch)):
            print(str(gp_batch[j] > bp_batch[j]), na_list[i+j], batch_goods[j], gp_batch[j], batch_bads[j], bp_batch[j], batch_masked[j].encode("utf8"), sep=u"\t")

        if i % 100 == 0:
            print(i, file=sys.stderr)
            sys.stdout.flush()

def read_gulordava():
    rows = csv.DictReader(open("generated.tab", encoding="utf8"), delimiter="\t")
    data = []
    for row in rows:
        row2 = next(rows)
        assert row["sent"] == row2["sent"]
        assert row["class"] == "correct"
        assert row2["class"] == "wrong"
        sent = row["sent"].lower().split()[:-1]
        good_form, bad_form = row["form"], row2["form"]
        sent[int(row["len_prefix"])] = "***mask***"
        data.append((" ".join(sent), row["n_attr"], good_form, bad_form))
    return data

def eval_gulordava():
    data = read_gulordava()
    masked_list, natt_list, good_list, bad_list = zip(*data)
    batch_size = 16  # Adjust this based on your hardware

    for i in range(0, len(masked_list), batch_size):
        batch_masked = masked_list[i:i+batch_size]
        batch_goods = good_list[i:i+batch_size]
        batch_bads = bad_list[i:i+batch_size]

        ps = get_probs_for_words_batch(batch_masked, batch_goods, batch_bads)
        if ps is None:
            continue
        gp_batch, bp_batch = ps
        for j in range(len(gp_batch)):
            print(str(gp_batch[j] > bp_batch[j]), natt_list[i+j], batch_goods[j], gp_batch[j], batch_bads[j], bp_batch[j], batch_masked[j].encode("utf8"), sep=u"\t")

        if i % 100 == 0:
            print(i, file=sys.stderr)
            sys.stdout.flush()

if "marvin" in sys.argv:
    eval_marvin()
elif "gul" in sys.argv:
    eval_gulordava()
elif "lgd" in sys.argv:
    eval_lgd()
else:
  print("There is no evaluation target")
