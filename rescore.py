import argparse
import csv
import json
import time
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from queue import PriorityQueue
from tqdm import tqdm

from evaluation import all_metrics, micro_f1
from datasets import load_full_codes
from icd_bert import ICDBertForMaskedLM, ICDBertTokenizer


MIMICDATA = "/data/cwhuang/projects/ijcai2020/data/mimicdata"


def rescore_made(args, dataset, scores, ind2c, c2ind, y_true, y_pred, y_pred_raw):
    made = torch.load(args.made_model)
    made.cuda()
    with open(args.made_vocab) as jsonfile:
        made_vocab = json.load(jsonfile)["rev_vocab"]
    
    y_pred_oracle = []
    y_pred_nbests = []
    for probs, pred, true in tqdm(zip(y_pred_raw, y_pred, y_true)):
        y_pred_nbest = get_n_best(probs, n=args.n_best)
        
        nbests = []
        nbest_labels = []
        for prob, labels in y_pred_nbest:
            nbest_labels.append(labels)
            f1 = micro_f1(labels, true)
            nbests.append((f1, prob, labels))

        labels_made = get_y_pred_made(nbest_labels, ind2c, made_vocab)
        made_losses = get_made_likelihood(args, made, made_vocab, labels_made)
        _, _, labels_oracle = sorted(nbests, key=lambda x:x[0])[-1]

        nbests_with_made = []
        for (f1, prob, labels), made_loss in zip(nbests, made_losses):
            nonzeros = len(np.nonzero(labels)[0])
            if nonzeros == 0: nonzeros = 1
            nbests_with_made.append((f1, -made_loss, nonzeros, prob, labels))
        y_pred_oracle.append(labels_oracle)
        y_pred_nbests.append(nbests_with_made)
        
    y_pred_oracle = np.array(y_pred_oracle)
    metrics = all_metrics(y_pred_oracle, y_true, k=[8, 15])
    print("oracle")
    print(metrics)

    rescore_grid(
        coefs=[0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0],
        alphas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        y_true=y_true,
        y_pred_nbests=y_pred_nbests
    )

    y_true_made, ids_made = get_y_true(dataset, made_vocab)
    y_true_losses = get_made_likelihood(args, made, made_vocab, y_true_made)
    print("y_true loss", np.mean(y_true_losses))
    y_pred_made = get_y_pred(scores, made_vocab, ids_made)
    y_pred_losses = get_made_likelihood(args, made, made_vocab, y_pred_made)
    print("y_pred loss", np.mean(y_pred_losses))


def rescore_bert(args, dataset, scores, ind2c, c2ind, y_true, y_pred, y_pred_raw):
    model = ICDBertForMaskedLM.from_pretrained(args.bert_model)
    model.eval()
    model.cuda()
    tokenizer = ICDBertTokenizer(os.path.join(args.bert_model, "vocab.txt"))

    y_pred_nbests = []
    for pred, pred_raw in tqdm(zip(y_pred, y_pred_raw)):
        y_pred_nbest = get_n_best(pred_raw, n=args.n_best)
        nbests_with_bert = []
        for prob, labels in y_pred_nbest:
            codes = [ind2c[ind] for ind in np.nonzero(labels)[0] if ind in ind2c]
            if len(codes) == 0:
                continue
            token_ids = tokenizer.convert_tokens_to_ids(codes)
            all_input_ids, all_attention_mask = [], []
            for i, token_id in enumerate(token_ids):
                input_ids, attention_mask = build_masked_input(
                    token_ids, tokenizer, mask_positions=[i]
                )
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
            
            all_input_ids = torch.tensor(all_input_ids).cuda()
            all_attention_mask = torch.tensor(all_attention_mask).cuda()
            with torch.no_grad():
                output = model(all_input_ids, all_attention_mask)
            prediction_scores = output[0]
            log_prob = 0.0
            for i, token_id in enumerate(token_ids):
                log_prob += prediction_scores[i, i].detach().log_softmax(dim=-1)[token_id].cpu().item()
            nbests_with_bert.append((0.0, log_prob, len(token_ids), prob, labels))
            
        y_pred_nbests.append(nbests_with_bert)

    with open("y_pred_nbests.bert", "wb") as pkl:
        pickle.dump(y_pred_nbests, pkl)

    rescore_grid(
        coefs=[0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0],
        alphas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        y_true=y_true,
        y_pred_nbests=y_pred_nbests
    )


def rescore_grid(coefs, alphas, y_true, y_pred_nbests):
    for coef in coefs:
        for alpha in alphas:
            print("coef: ", coef, ", alpha: ", alpha)
            this_pred = []
            for nbests in y_pred_nbests:
                _, _, _, _, labels = sorted(nbests, key=lambda x:x[3]+coef*x[1]/(x[2]**alpha))[-1]
                this_pred.append(labels)
            this_pred = np.array(this_pred)
            metrics = all_metrics(this_pred, y_true, k=[8, 15])
            print(metrics)
            return this_pred


def build_masked_input(input_ids, tokenizer, mask_positions=None):
    attention_mask = [1] * len(input_ids)
    if mask_positions:
        if isinstance(mask_positions, int):
            mask_positions = [mask_positions]
        input_ids = [tokenizer.mask_token_id if i in mask_positions else ind for i, ind in enumerate(input_ids)]
    return input_ids, attention_mask


def main(args):
    with open(args.dataset) as csvfile:
        reader = csv.DictReader(csvfile)
        dataset = list(reader)

    with open(args.scores) as jsonfile:
        scores = json.load(jsonfile)

    ind2c, c2ind = get_code_vocab(args)
    vocab_size = len(ind2c)
    y_true, ids = get_y_true(dataset, c2ind)
    y_pred = get_y_pred(scores, c2ind, ids)
    y_pred_raw = get_y_pred_raw(scores, c2ind, ids)
    metrics = all_metrics(y_pred, y_true, k=[8, 15], yhat_raw=y_pred_raw)    
    print(metrics)

    if args.rescorer == "made":
        rescore_made(args, dataset, scores, ind2c, c2ind, y_true, y_pred, y_pred_raw)
    elif args.rescorer == "bert":
        rescore_bert(args, dataset, scores, ind2c, c2ind, y_true, y_pred, y_pred_raw)
   

def get_y_pred_hard(c2ind, ids):
    predictions = dict()
    for line in open("predictions/CAML_mimic2_full/preds_test.psv"):
        line = line.strip().rstrip('|').split('|')
        id = line[0]
        labels = [] if len(line) == 1 else line[1:]
        predictions[id] = labels
    preds = np.zeros((len(predictions), len(c2ind)))
    for i, (id, labels) in enumerate(predictions.items()):
        for label in labels:
            if label in c2ind:
                preds[ids[id], c2ind[label]] = 1
    return preds


def get_n_best(probs, n=10):
    flip_idx = np.argsort(np.abs(probs-0.5))[:n]
    labels = np.zeros(len(probs))
    cum_prob = 0.0
    for i, prob in enumerate(probs):
        if i in flip_idx:
            continue
        if prob >= 0.5:
            labels[i] = 1
            cum_prob += np.log(prob)
        else:
            labels[i] = 0
            cum_prob += np.log(1-prob)

    last_queue = PriorityQueue()
    last_queue.put((cum_prob, labels))
    for i, prob in enumerate(probs):
        if i not in flip_idx:
            continue
        queue = PriorityQueue()
        for cum_prob, labels in last_queue.queue:
            labels2 = np.copy(labels)
            labels[i] = 1
            queue.put((cum_prob + np.log(prob+1e-6), labels))
            labels2[i] = 0
            queue.put((cum_prob + np.log(1-prob+1e-6), labels2))

        while len(queue.queue) > n:
            _ = queue.get()

        last_queue = queue

    n_best = []
    while not queue.empty():
        n_best.append(queue.get())
    n_best = n_best[::-1]
    return n_best


def get_made_likelihood(args, made, made_vocab, y_matrix):
    made.eval()
    batch_size = 100
    n_examples, dim_input = y_matrix.shape
    nsteps = n_examples // batch_size
    if n_examples % batch_size != 0:
        nsteps += 1
    losses = []
    for step in range(nsteps):
        if step == nsteps - 1:
            x = torch.tensor(y_matrix[step*batch_size:])
        else:
            x = torch.tensor(y_matrix[step*batch_size:(step+1)*batch_size])

        x = x.float().cuda()
        x_pred = torch.zeros_like(x)
        for s in range(args.n_samples):
            made.update_masks()
            with torch.no_grad():
                x_pred += made(x)
        x_pred /= args.n_samples
        loss = F.binary_cross_entropy_with_logits(x_pred, x, reduction='none')
        loss = loss.sum(dim=-1).cpu().numpy()
        losses.append(loss)
    losses = np.concatenate(losses, axis=0)

    return losses


def get_y_true(dataset, c2ind):
    labels = np.zeros((len(dataset), len(c2ind)))
    ids = dict()
    for i, entry in enumerate(dataset):
        ids[entry["HADM_ID"]] = i
        for label in entry["LABELS"].strip().split(";"):
            if label in c2ind:
                labels[i, c2ind[label]] = 1
    return labels, ids


def get_y_pred(scores, c2ind, ids):
    preds = np.zeros((len(scores), len(c2ind)))
    for i, (id, entry) in enumerate(scores.items()):
        for label, prob in entry.items():
            if prob >= 0.5 and label in c2ind:
                preds[ids[id], c2ind[label]] = 1
    return preds


def get_y_pred_made(y_pred, ind2c, c2ind):
    preds = np.zeros((len(y_pred), len(c2ind)))
    for i, pred in enumerate(y_pred):
        for label in np.nonzero(pred)[0]:
            label_name = ind2c[label]
            if label_name in c2ind:
                preds[i, c2ind[label_name]] = 1
    return preds


def get_y_pred_raw(scores, c2ind, ids):
    preds = np.zeros((len(scores), len(c2ind)))
    for i, (id, entry) in enumerate(scores.items()):
        for label, prob in entry.items():
            if label in c2ind:
                preds[ids[id], c2ind[label]] = prob
    return preds


def get_code_vocab(args):
    if args.code_type == "full":
        ind2c, _ = load_full_codes(
            f"{MIMICDATA}/{args.version}/train_{args.code_type}.csv", version=args.version)
    else:
        codes = set()
        with open(f"{MIMICDATA}/{args.version}/TOP_{args.code_type}_CODES.csv") as labelfile:
            lr = csv.reader(labelfile)
            for i,row in enumerate(lr):
                codes.add(row[0])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
    c2ind = {c:i for i,c in ind2c.items()}
    return ind2c, c2ind


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="processed dataset file")
    parser.add_argument("scores", help="predicted scores")
    parser.add_argument("rescorer", choices=["made", "bert", "xlnet"], help="rescorer model")
    parser.add_argument("--made_model", help="path to pre-trained made model")
    parser.add_argument("--made_vocab", help="path to the vocabulary file of pre-trained made model")
    parser.add_argument("--bert_model", help="path to pre-trained bert model")
    parser.add_argument("--code_type", default="full", choices=["full", "50"], help="code set")
    parser.add_argument("--version", default="mimic3", choices=["mimic3", "mimic2"],
                        help="path to data used for pre-training made")
    parser.add_argument("--n_samples", default=10, type=int, help="number of masks MADE samples during inference")
    parser.add_argument("--n_best", default=10, type=int, help="number of predictions to be rescored")
    args = parser.parse_args()
    main(args)
