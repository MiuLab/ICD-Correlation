ICDRescore
===

## Pre-training
run `run_pretraining.sh`

## Rescoring
1. modify the **MIMICDATA** path in `rescore.py`
2. generate the prediction file with **persistence.write_preds** method from CAML, which you should have already got after training a CAML model
3. run rescoring
  * For BERT
    ```bash
    python3 rescore.py \
      --n_best 50 \
      --bert_model models/bert-icd-pretrained \
      --version mimic3 \
      ../data/mimicdata/mimic3/test_full.csv \
      predictions/CAML_mimic3_full/pred_100_scores_test.json \
      bert
    ```
  * For MADE
    ```bash
    python3 rescore.py \
      --n_best 50 \
      --made_model mimic3.model \
      --made_vocab mimic3.vocab.json \
      ../data/mimicdata/mimic3/test_full.csv \
      predictions/CAML_mimic3_full/pred_100_scores_test.json \
      made
    ```
