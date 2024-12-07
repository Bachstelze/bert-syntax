# coding=utf-8
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, BertTokenizer
import torch
import sys
import csv
import logging
import itertools

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from pprint import pprint
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name2 = "openai-community/gpt2"
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)

model2 = AutoModelForCausalLM.from_pretrained(model_name2)
model2.eval()
model2.to(device)


tokenizer2.pad_token_id = tokenizer2.eos_token_id
# avoid warning
model2.generation_config.pad_token_id = tokenizer2.pad_token_id

starting_text = "the author knows many different foreign languages and"

def get_gpt_word_prob(input_sequence, word, model, tokenizer, verbose=False):
  tokenized_inputs = tokenizer([input_sequence], return_tensors="pt").to(device)
  input_length = tokenized_inputs.input_ids.shape[1]
  #print(input_length)
  force_word_ids = tokenizer([word], add_special_tokens=False).input_ids
  #print(force_word_ids[0])
  force_word_ids_length = len(force_word_ids[0])
  #print(force_word_ids_length)
  outputs = model.generate(**tokenized_inputs,
                         num_beams=2,
                         force_words_ids=force_word_ids, 
                         return_dict_in_generate=True,
                         output_scores=True,
                         max_new_tokens=force_word_ids_length)
  #print(outputs.sequences)
  transition_scores = model.compute_transition_scores(
    outputs.sequences, outputs.scores, normalize_logits=True
    )
  #print(transition_scores)
  generated_tokens = outputs.sequences[:, input_length:]
  #print(generated_tokens)

  word_probability = 0.0
  for tok, score in zip(generated_tokens[0], transition_scores[0]):
    if verbose:
      print("| token | token string | log probability | probability")
      print(f"| {tok:5d} | {tokenizer.decode(tok):8s} | {score.numpy():.3f} | {np.exp(score.numpy()):.2%}")
    word_probability += score.cpu().numpy()
  return word_probability

model_name = "openai-gpt"
print("using model: {}".format(model_name), file=sys.stderr)

split_words = True
if 'no_split' in sys.argv:
    split_words = False
    print("We don't split words", file=sys.stderr)

use_postfix = False
if 'use_postfix' in sys.argv:
    use_postfix = True
    print("We compute probabilities over the entire sentence", file=sys.stderr)

model = OpenAIGPTLMHeadModel.from_pretrained(model_name)
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
model.eval()
model.to(device)


def get_probs_for_words(sent, w1, w2):
    pre, target, post = sent.split("***")
    if "mask" in target.lower():
        target = ["[MASK]"]
    else:
        target = tokenizer.tokenize(target)
    tokens = tokenizer.tokenize(pre)
    target_idx = len(tokens)

    # Filter answers based on BERT wordpieces to align with BERT results
    try:
        word_ids=bert_tokenizer.convert_tokens_to_ids([w1,w2])
    except KeyError:
        print("skipping",w1,w2,"bad wins")
        return None

    tok_w1, tok_w2 = tokenizer.tokenize(w1), tokenizer.tokenize(w2)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    w1_ids = tokenizer.convert_tokens_to_ids(tok_w1)
    w2_ids = tokenizer.convert_tokens_to_ids(tok_w2)
    if len(input_ids) == 0:
        print("skipping",pre,w1,w2,"empty beggingin")
        return None

    if not split_words and (len(tok_w1) > 1 or len(tok_w2) > 1):
        print("skipping",pre,w1,w2,"splitted words")
        return None

    if use_postfix:
        # Add post focus tokens
        end_tokens = tokenizer.tokenize(post)
        end_ids = tokenizer.convert_tokens_to_ids(end_tokens)
        w1_ids += end_ids
        w2_ids += end_ids

    # Compute the score for w1 and w2
    score_w1 = get_gpt_word_prob(pre, w1, model2, tokenizer2)
    score_w2 = get_gpt_word_prob(pre, w2, model2, tokenizer2)
    """
    add_tok_w1 = []
    add_tok_w2 = []
    score_w1 = 0
    score_w2 = 0
    for ids_w1, ids_w2 in itertools.zip_longest(w1_ids, w2_ids):
        tens = torch.LongTensor([input_ids + add_tok_w1, input_ids + add_tok_w2]).to(device)
        with torch.no_grad():
            res = model(tens)
            res = res[..., 0:model.config.vocab_size] # Restrict to the vocabulary only
            res = torch.nn.functional.log_softmax(res, dim=-1)
        if ids_w1 is not None:
            score_w1 = score_w1 + res[0, -1, ids_w1].item()
        if ids_w2 is not None:
            score_w2 = score_w2 + res[1, -1, ids_w2].item()
        add_tok_w1.append(ids_w1 if ids_w1 is not None else [0])
        add_tok_w2.append(ids_w2 if ids_w2 is not None else [0])

    # Compute the score for w2
    # add_tok = []
    # score_w2 = 0
    # for ids in w2_ids:
    #     tens = torch.LongTensor(input_ids + add_tok).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         res = model(tens)
    #         res = res[..., 0:model.config.vocab_size] # Restrict to the vocabulary only
    #         res = torch.nn.functional.log_softmax(res,dim=-1)
    #     score_w2 = score_w2 + res[0, -1, ids]
    #     add_tok.append(ids)
    """
    return [float(score_w1), float(score_w2)]


from collections import Counter


def load_marvin():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
    out = []
    for line in open("marvin_linzen_dataset.tsv"):
        case = line.strip().split("\t")
        cc[case[1]] += 1
        g, ug = case[-2], case[-1]
        g = g.split()
        ug = ug.split()
        assert len(g) == len(ug), (g, ug)
        diffs = [i for i, pair in enumerate(zip(g, ug)) if pair[0] != pair[1]]
        if len(diffs) != 1:
            # print(diffs)
            # print(g,ug)
            continue
        assert len(diffs) == 1, diffs
        gv = g[diffs[0]]  # good
        ugv = ug[diffs[0]]  # bad
        g[diffs[0]] = "***mask***"
        g.append(".")
        out.append((case[0], case[1], " ".join(g), gv, ugv))
    return out


def eval_marvin():
    o = load_marvin()
    print(len(o), file=sys.stderr)
    from collections import defaultdict
    import time

    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    for i, (case, tp, s, g, b) in enumerate(o):
        ps = get_probs_for_words(s, g, b)
        if ps is None:
            ps = [0, 1]
        gp = ps[0]
        bp = ps[1]
        print(gp > bp, case, tp, g, b, s)
        if i % 100 == 0:
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()


def eval_lgd():
    for i, line in enumerate(open("lgd_dataset.tsv", encoding="utf8")):
        #    for i,line in enumerate(open("lgd_dataset_with_is_are.tsv",encoding="utf8")):
        na, _, masked, good, bad = line.strip().split("\t")
        ps = get_probs_for_words(masked, good, bad)
        if ps is None:
            continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp > bp), na, good, gp, bad, bp, masked.encode("utf8"), sep=u"\t")
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
        sent = row["sent"].lower().split()[:-1]  # dump the <eos> token.
        good_form = row["form"]
        bad_form = row2["form"]
        sent[int(row["len_prefix"])] = "***mask***"
        sent = " ".join(sent)
        data.append((sent, row["n_attr"], good_form, bad_form))
    return data


def eval_gulordava():
    for i, (masked, natt, good, bad) in enumerate(read_gulordava()):
        if good in ["is", "are"]:
            print("skipping is/are")
            continue
        ps = get_probs_for_words(masked, good, bad)
        if ps is None:
            continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp > bp), natt, good, gp, bad, bp, masked.encode("utf8"), sep=u"\t")
        if i % 100 == 0:
            print(i, file=sys.stderr)
            sys.stdout.flush()


if "marvin" in sys.argv:
    eval_marvin()
elif "gul" in sys.argv:
    eval_gulordava()
else:
    eval_lgd()
