ICDRescore
===
[Paper](https://aclanthology.org/2021.naacl-main.318/)
| [Presentation](https://underline.io/lecture/19622-modeling-diagnostic-label-correlation-for-automatic-icd-coding)

Source code for our NAACL 2021 paper *Modeling Diagnostic Label Correlation for Automatic ICD Coding*.

## Pre-training
1. preprocess mimic dataset with the scripts from [caml-mimic](https://github.com/jamesmullenbach/caml-mimic)
2. prepare pretraining data by extracting the labels from `*_{full,50}.csv` and making the labels **space-separated**, one line for each training instance.
3. run `run_pretraining.sh`

## Rescoring
1. modify the **MIMICDATA** path in `rescore.py`
2. If you're using MultiResCNN or LAAT, you'll need to generate the prediction file with the **persistence.write_preds** method from CAML.
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


## Reference
If you find our work useful, please cite the following paper
  
    @inproceedings{tsai2021modeling,
      title={Modeling Diagnostic Label Correlation for Automatic ICD Coding},
      author={Tsai, Shang-Chi and Huang, Chao-Wei and Chen, Yun-Nung},
      booktitle={Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
      pages={4043--4052},
      year={2021}
    }
