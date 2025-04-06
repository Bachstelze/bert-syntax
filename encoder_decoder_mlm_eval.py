from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, AutoModelForCausalLM, BertTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
import sys
import csv
import logging
import itertools
import numpy as np
from collections import Counter, defaultdict
import time

script, dataset_type, model_name, batch_size, filter_tokens, split_words, use_postfix = sys.argv
# convert the batch size string
batch_size = int(batch_size)

# look for keywords
split_words = 'no_split' not in sys.argv
# not supported yet
use_postfix = 'use_postfix' in sys.argv
if use_postfix:
    print("postfix is not yet compatible with mask padding", file=sys.stderr)
    sys.exit()

print("using split words: {}".format(split_words), file=sys.stderr)
print("using postfix: {}".format(use_postfix), file=sys.stderr)
print("using filtered tokens: {}".format(filter_tokens), file=sys.stderr)
print("using batch size: {}".format(batch_size), file=sys.stderr)
print("using model: {}".format(model_name), file=sys.stderr)

"""
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
"""
instruction_base_model = "FacebookAI/roberta-base"
config = AutoConfig.from_pretrained(instruction_base_model)  # or any other suitable Bert variant
# Create a new AutoModelForMaskedLM instance
masked_lm_model = AutoModelForMaskedLM.from_pretrained(
    model_name,
    config=config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Create the pipeline with the specified tokenizer
bert_unmask = pipeline('fill-mask', model=masked_lm_model, tokenizer=tokenizer)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
masked_lm_model.eval()
masked_lm_model.to(device)

# Use the padding token
padding_token_id = tokenizer.pad_token_id
mask_token = tokenizer.mask_token

def get_full_token_score(input, target_token, processed_token, overall_token_score, sub_token_count):
    processed_token = ""
    pipe_output = bert_unmask(input, targets=[target_token])
    print(pipe_output)
    predicted_token_score = pipe_output[0]["score"]
    overall_token_score += predicted_token_score
    sub_token_count += 1
    predicted_token = pipe_output[0]["token_str"]
    processed_token += predicted_token
    new_target_token = target_token[len(predicted_token):]
    combined_input = input.replace(mask_token, predicted_token+mask_token)
    if processed_token == target_token:
        #print("final score")
        finall_score = overall_token_score/sub_token_count
        #print(finall_score)
        return finall_score


    #print(combined_input)
    #print("combine subtoken, recusrive call")
    return get_full_token_score(input, target_token, processed_token, overall_token_score, sub_token_count)
  
def get_probs_for_words_mlm(sentence, w1, w2):
  print(sentence)
  pre, target, post = sentence[0].split("***")
  pipeline_input = pre + mask_token + post
  print(pipeline_input)

  score_w1 = get_full_token_score(pipeline_input, w1[0], "", 0.0, 0)
  score_w2 = get_full_token_score(pipeline_input, w2[0], "", 0.0, 0)
  
  return score_w1, score_w2

  
def get_probs_for_words_batch(sentences, w1_list, w2_list):
    encoder_input_ids_batch = []
    input_ids_batch = []
    encoder_attention_mask_batch = []
    decoder_attention_mask_batch = []
    w1_ids_batch = []
    w2_ids_batch = []

    for sent, w1, w2 in zip(sentences, w1_list, w2_list):
        pre, target, post = sent.split("***")
        # we set directly the target token
        # target = ["[MASK]"] if "mask" in target.lower() else tokenizer.tokenize(target)

        tokens = tokenizer.tokenize(pre)
        target_idx = len(tokens)

        # Filter answers based on BERT wordpieces to align with BERT results
        if bool(filter_tokens):
            try:
                word_ids = bert_tokenizer.convert_tokens_to_ids([w1, w2])
            except KeyError:
                print("skipping", w1, w2, "bad tokens")
                print("skipping", w1, w2, "bad tokens", file=sys.stderr)
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

        decoder_attention_mask = [1] * len(input_ids)
        assert(len(input_ids) == len(decoder_attention_mask))
        decoder_attention_mask_batch.append(decoder_attention_mask)

        # create the encoder input
        encoder_input = str(pre) + mask_token + str(post)
        encoder_tokens = tokenizer.tokenize(encoder_input)
        encoder_input_ids = tokenizer.convert_tokens_to_ids(encoder_tokens)
        encoder_input_ids_batch.append(encoder_input_ids)

        # create the attention mask to the encoder input
        attention_mask = [1] * len(encoder_tokens)
        encoder_attention_mask_batch.append(attention_mask)

    if not input_ids_batch:
        print("no input batch", file=sys.stderr)
        return None

    # Pad the sequences to the max length in the encoder batch
    max_len_encoder = max(len(ids) for ids in encoder_input_ids_batch)
    padded_encoder_input_ids_batch = []
    padded_attention_mask_batch = []

    for encoder_input_ids, encoder_attention_mask in zip(encoder_input_ids_batch, encoder_attention_mask_batch):
        padded_encoder_input_ids = encoder_input_ids + [padding_token_id] * (max_len_encoder - len(encoder_input_ids))
        padded_encoder_input_ids_batch.append(padded_encoder_input_ids)

        padded_attention_mask = encoder_attention_mask + [0] * (max_len_encoder - len(encoder_input_ids))
        padded_attention_mask_batch.append(padded_attention_mask)

    # Pad the sequences to the max length in the decoder batch
    max_len_decoder = max(len(ids) for ids in input_ids_batch)
    padded_input_ids_batch = []
    padded_decoder_attention_mask_batch = []
    padded_w1_ids_batch = []
    padded_w2_ids_batch = []

    for input_ids, w1_ids, w2_ids, dec_attn_mask in zip(input_ids_batch, w1_ids_batch, w2_ids_batch, decoder_attention_mask_batch):
        padded_input_ids = input_ids + [padding_token_id] * (max_len_decoder - len(input_ids))
        padded_w1_ids = w1_ids + [padding_token_id] * (max_len_decoder - len(w1_ids))
        padded_w2_ids = w2_ids + [padding_token_id] * (max_len_decoder - len(w2_ids))
        assert(len(dec_attn_mask) == len(input_ids))
        padded_decoder_attention_mask = dec_attn_mask + [0] * (max_len_decoder - len(dec_attn_mask))
        padded_input_ids_batch.append(padded_input_ids)
        padded_decoder_attention_mask_batch.append(padded_decoder_attention_mask)
        padded_w1_ids_batch.append(padded_w1_ids)
        padded_w2_ids_batch.append(padded_w2_ids)

    # Convert lists to tensors
    tens_encoder_input = torch.LongTensor(padded_encoder_input_ids_batch).to(device)
    tens_encoder_attention_mask = torch.LongTensor(padded_attention_mask_batch).to(device)
    tens_input = torch.LongTensor(padded_input_ids_batch).to(device)
    tens_decoder_attention_mask = torch.LongTensor(padded_decoder_attention_mask_batch).to(device)
    tens_w1 = torch.LongTensor(padded_w1_ids_batch).to(device)
    tens_w2 = torch.LongTensor(padded_w2_ids_batch).to(device)

    with torch.no_grad():
        res = model(attention_mask = tens_encoder_attention_mask, input_ids=tens_encoder_input, decoder_input_ids=tens_input, decoder_attention_mask = tens_decoder_attention_mask)[0]

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
    for line in open("../../bert-syntax/marvin_linzen_dataset.tsv"):
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

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]
        batch_goods = goods[i:i+batch_size]
        batch_bads = bads[i:i+batch_size]
        if bool(filter_tokens):
          try:
              word_ids = bert_tokenizer.convert_tokens_to_ids([batch_goods, batch_bads])
          except KeyError:
              print("skipping", w1, w2, "bad tokens")
              print("skipping", w1, w2, "bad tokens", file=sys.stderr)
              continue

        ps = get_probs_for_words_mlm(batch_sentences, batch_goods, batch_bads)
        if ps is None:
            continue
        gp_batch, bp_batch = ps
        print(gp_batch > bp_batch, cases[i], types[i], batch_goods, batch_bads, batch_sentences)
        """
        for j in range(len(gp_batch)):
            print(gp_batch[j] > bp_batch[j], cases[i+j], types[i+j], batch_goods[j], batch_bads[j], batch_sentences[j])
        """

        if i % 100 == 0:
            print(i, time.time() - start, file=sys.stderr)
            start = time.time()
            sys.stdout.flush()

def eval_lgd():
    lines = [line.strip().split("\t") for line in open("../../bert-syntax/lgd_dataset.tsv", encoding="utf8")]
    na_list, masked_list, good_list, bad_list = zip(*[(line[0], line[2], line[3], line[4]) for line in lines])

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
    rows = csv.DictReader(open("../../bert-syntax/generated.tab", encoding="utf8"), delimiter="\t")
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
    print("eval marvin", file=sys.stderr)
    eval_marvin()
elif "gul" in sys.argv:
    print("eval gulordava", file=sys.stderr)
    eval_gulordava()
else:
    print("the dataset of lgd is missing", file=sys.stderr)
    #eval_lgd()
