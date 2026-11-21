# Member 1: text NLP and ASR

This package is the textual sarcasm and irony module, plus speech-to-text, for SarcasTone. All of Member 1’s code lives in `member1_text/`. The repository root `README.md`, `hello.py`, and `SarcasTone/` belong to the shared project and are left unchanged.

Stage 1 is the scaffold: Python 3.11 check, dependencies, configuration, and this document. Loaders, the trainer, ASR, evaluation, Captum, and the four public functions are specified here and are added in later stages. No accuracy number, checkpoint, or dataset row is included, because none has been produced by a training run.

## Runtime

The package requires **Python 3.11**. Importing `member1_text` does not exit. Check the interpreter with the CLI, or call `require_python_311()` before training:

```bash
python -m member1_text
python -c "import member1_text; member1_text.require_python_311()"
```

On any other version the CLI writes this message to stderr and exits 1. `require_python_311()` raises `RuntimeError` with the same text:

```text
member1_text requires Python 3.11. This interpreter is Python X.Y.Z.
Install Python 3.11 and run this package with that interpreter.
```

ffmpeg must be on `PATH`. The trainer uses it to turn non-wav audio into 16 kHz mono wav. ffmpeg is a system package, not a pip package. `librosa` is not a dependency.

## Install

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r member1_text/requirements.txt
export PYTHONPATH=member1_text/src
```

Confirm the interpreter before importing:

```bash
python -c "import sys; assert sys.version_info[:2] == (3, 11); import member1_text; print(member1_text.__version__)"
```

The first fine-tune downloads `roberta-base` from Hugging Face. If that download fails, training stops and writes no checkpoint.

## Layout

```text
member1_text/
  README.md
  requirements.txt
  config.yaml             default configuration; kept at this path
  .gitignore
  colab/                  train_colab.py is added with the Colab stage
  configs/                extra configuration files in later stages
  notebooks/
  datasets/               dataset-specific notes and helpers
  preprocessing/
  asr/
  models/
  training/
  evaluation/
  inference/
  explainability/
  checkpoints/            local checkpoint placeholder
  data/                   local corpora, ignored by git
    mustard/
    mustard_pp/
    sarc/main/
    semeval2018/
  src/member1_text/       package; data/ holds one loader per corpus
  tests/
  outputs/
    provenance/           hash and license notes that may be committed
    eval/                 metrics, ignored by git
    checkpoints/          model weights, ignored by git
    mlruns/               MLflow, ignored by git
    plots/                figures, ignored by git
```

`outputs/` ignores data, wavs, checkpoints, MLflow runs, and plots. Eval JSON stays available to commit after a real run. A provenance file for the public MUStARD JSON hash may be committed. Corpus text for SARC, SemEval-2018, and MUStARD++ is never committed.

## Configuration

`config.yaml` is the default set for model, training, reporting, deployment, ASR, and local dataset paths. Later stages read it. The primary model is `roberta-base`. `microsoft/deberta-v3-base` is the optional comparison and is trained only when a command passes `--model deberta-v3-base`. Checkpoints from the two models are not interchangeable.

## Datasets

Each corpus has its own loader, schema, and label meaning. They are not concatenated for the primary run. A cross-corpus experiment is a separate MLflow run and only after MUStARD / MUStARD++ overlap removal. If a required local file is missing, that dataset’s command exits with the source URL and writes nothing that looks like a result.

| Dataset | Local path | Source | License note |
| --- | --- | --- | --- |
| MUStARD | `member1_text/data/mustard/sarcasm_data.json` | [soujanyaporia/MUStARD](https://github.com/soujanyaporia/MUStARD) `data/sarcasm_data.json`. Utterance clips: [MichiganNLP/MUStARD](https://huggingface.co/datasets/MichiganNLP/MUStARD) `mmsd_raw_data.zip`, `utterances_final/<id>.mp4` | MIT for the GitHub text release |
| MUStARD++ | `member1_text/data/mustard_pp/mustard++_text.csv` | [cfiltnlp/MUStARD_Plus_Plus](https://github.com/cfiltnlp/MUStARD_Plus_Plus) | No license file in the GitHub repo. Research release. The loader reads a local CSV and does not scrape Drive |
| SARC 2.0 | `member1_text/data/sarc/main/` with `train-balanced.csv.bz2`, `test-balanced.csv.bz2`, `comments.json.bz2` | [nlp.cs.princeton.edu/old/SARC/2.0](https://nlp.cs.princeton.edu/old/SARC/2.0/) | Citation required. The corpus has no SPDX license. A third-party mirror is not the download source. The official balanced files are the split and are not reshuffled |
| SemEval-2018 Task 3 | `member1_text/data/semeval2018/` cleaned Task A and Task B train files, and the official test file only when it is already local | Task organizers. Use `SemEval2018-T3-train-taskA.txt` and the Task B pair | CC BY-NC-SA 4.0, academic use, not redistributed here. Tweet text is not written into the repo. `ironyHashtags` files are rejected because those hashtags leak the label |

Label meaning stays with the dataset. SemEval Task A is irony, not sarcasm. SemEval Task B stays four-way. MUStARD++ explicit emotion, valence, and arousal are separate tasks and are not folded into the sarcasm bit.

MUStARD splits, seed 42:

- stratified 5-fold on the sarcasm label
- speaker-grouped split so each speaker is on only one side
- `data/split_indices.p` only when that pickle loads on Python 3.11. If it does not, the split is reported as unavailable

## Cross-validation and independent test

Five-fold results and independent test results are different reports. They are written to different files and they answer different questions.

For the primary MUStARD sarcasm run, stratified 5-fold cross-validation trains on four folds and validates on the held-out fold. Each fold’s score is a **validation** score. The run writes:

`member1_text/outputs/eval/mustard__sarcasm/cv_validation.json`

That file contains:

- `role: cross_validation_validation`
- `is_independent_test: false`
- the validation metrics of all five folds
- the mean and standard deviation of validation macro-F1 across folds
- `best_fold_validation_macro_f1`, with `used_for_epoch_selection_only: true` and `reported_as_test: false`

The best validation fold is a model-selection statistic. It is not an independent test result. It is not copied into `independent_test.json`, it is not labeled `test_macro_f1`, and it is not described as held-out test performance.

`independent_test.json` is written only for a split that was not used to choose the epoch count or the weights:

- SARC official `test-balanced`, after early stopping on a held-out slice of the official train file
- SemEval official gold test, and only when that gold file is present locally

A slice carved from SemEval train is `train_holdout.json`. It is not the official test. The MUStARD speaker-grouped split is `speaker_grouped.json`. It is a speaker-shift check, not the five-fold score and not an independent test score for the deployment checkpoint. No single file is presented as performance on “the data.”

## Deployment checkpoint

MUStARD’s JSON has no official test set, so the checkpoint other members load is a **full-data refit**, not the weights of the best validation fold.

1. Run stratified 5-fold cross-validation with seed 42. Early stopping watches validation macro-F1 with patience 2. Record, for each fold, the epoch that maximized that fold’s validation macro-F1.
2. Set `deployment_epochs` to the median of those five epoch numbers. With five folds the median is the middle value after sorting.
3. Start again from `roberta-base`. Train on **all** labeled MUStARD examples for exactly `deployment_epochs`. This refit does not use a validation split and does not early-stop. Class weights are computed on the full set. The seed is 42.
4. Save that run to `member1_text/outputs/checkpoints/mustard__sarcasm__roberta-base/`.
5. Write `deployment_refit.json` with `role: deployment_refit`, the five fold epochs, `deployment_epochs`, and the path of `cv_validation.json`. This file has no test metric. The cross-validation mean may be cited only as the CV validation estimate, next to the CV file, never as the refit’s test score.
6. Leave fold weights in the cross-validation run directory. `copy_best_fold_weights` is false.

`config.yaml` records the same rule under `evaluation` and `deployment`.

DeBERTa-v3 repeats the MUStARD five-fold protocol only when requested, and its deployment refit is a different directory. It is not written under the RoBERTa checkpoint name.

## Train

These commands are the interface later stages implement. They are not runnable in Stage 1, because `train.py` is not in the tree yet.

Primary cross-validation, from the repository root with `PYTHONPATH=member1_text/src`:

```bash
python -m member1_text.train \
  --dataset mustard \
  --task sarcasm \
  --model roberta-base \
  --split stratified_5fold
```

Deployment refit, after that cross-validation has written `cv_validation.json`:

```bash
python -m member1_text.train \
  --dataset mustard \
  --task sarcasm \
  --model roberta-base \
  --split deployment_refit
```

Other real runs are separate MLflow experiments: MUStARD speaker-grouped, MUStARD++ sarcasm, MUStARD++ explicit emotion, SARC official balanced train with a train-side early-stopping slice, then one evaluation on the official balanced test, and SemEval Task A and Task B when the local files exist. Optional comparison:

```bash
python -m member1_text.train \
  --dataset mustard \
  --task sarcasm \
  --model deberta-v3-base \
  --split stratified_5fold
```

Defaults from `config.yaml`: AdamW, learning rate `2e-5`, weight decay `0.01`, warmup ratio `0.06`, batch size 16 on GPU and 4 on CPU, at most 6 epochs, gradient clip `1.0`. CUDA is used when `torch.cuda.is_available()` is true. `--device cpu` forces CPU. Mixed precision uses `torch.autocast` and `GradScaler` on CUDA (bfloat16 when the GPU supports it, otherwise float16). CPU training stays float32. Class weights are inverse frequency on the training split only.

MLflow tracking URI: `member1_text/outputs/mlruns`. Raw SemEval and SARC text is not logged. ASR transcripts are not the default training text. A later run may train or evaluate on Whisper text and is tagged `text_source=asr`.

## Resume

Each finished epoch writes `checkpoint-<epoch>/` with `model.save_pretrained`, `tokenizer.save_pretrained`, and `trainer_state.pt` (optimizer, scheduler, epoch, best score, class weights, config). Continue from that directory:

```bash
python -m member1_text.train \
  --resume member1_text/outputs/checkpoints/<run>/checkpoint-<epoch>
```

No checkpoint is created until a real training run finishes an epoch.

## Evaluate, explain, transcribe

```bash
python -m member1_text.evaluate \
  --checkpoint member1_text/outputs/checkpoints/mustard__sarcasm__roberta-base \
  --dataset mustard \
  --task sarcasm

python -m member1_text.explain \
  --checkpoint member1_text/outputs/checkpoints/mustard__sarcasm__roberta-base \
  --text "Oh, great, another meeting." \
  --context "We already sat through two."

python -m member1_text.asr \
  --audio path/to/utterance.mp4
```

Evaluation writes accuracy, precision, recall, macro-F1, and weighted-F1, plus a confusion matrix. Binary tasks also write ROC-AUC and PR-AUC from the positive-class probability. Regression tasks write MSE and MAE. Plots are the confusion matrix, ROC, precision-recall, and train versus validation loss. False-positive and false-negative lists for SemEval store ids only.

Captum Integrated Gradients runs on a saved RoBERTa checkpoint, over the token embedding layer. Attributions are summed to tokenizer tokens, then to whitespace words, and written as HTML. Limits, which the explain command repeats: the baseline is the pad embedding, attributions are not causal, subword pieces are not words, context tokens and target tokens can both receive mass, and a highlight is not a sarcasm reason. Captum is not run on the TF-IDF baseline.

`transcribe_audio` uses faster-whisper, default model `small`, `language=en`, beam size 5. CUDA uses float16 when a GPU is visible. CPU uses int8. Accepted suffixes are `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, and `.mp4`. Missing ffmpeg or an unknown suffix is an error. The result includes text, language, language probability, duration, segments, model name, device, compute type, and `text_source="asr"`. Gold text stays `text_source="gold"`. ASR text is stored beside it. Word error rate uses `jiwer` when both strings exist. Sarcasm metrics on gold text and on ASR text are two rows.

## Colab

`member1_text/colab/train_colab.py` will be a normal Python script. It checks Python 3.11, installs `requirements.txt`, selects CUDA when Colab provides a GPU, and calls `member1_text.train` with the dataset, task, and an output directory on Drive or `/content`. It does not contain a second training loop and it does not embed scores.

## Checkpoint layout

A completed checkpoint directory contains:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- tokenizer files from `tokenizer.save_pretrained`
- `trainer_state.pt`

The deployment directory is `checkpoints/<dataset>__<task>__<model>/`. Tokenizer files are part of the checkpoint.

## Embeddings and the public API

`member1_text.api` loads the deployment RoBERTa sarcasm checkpoint:

- `encode_text(text, context=None)` returns `input_ids`, `attention_mask`, `context_mask`, `target_mask`
- `predict_text(text, context=None)` returns `label`, `probability_sarcasm`, `logits`
- `get_text_embedding(text, context=None)` returns `utterance` float32 `(768,)`, `context` float32 `(768,)` or a zero vector with `has_context=False`, `fused` float32 `(2305,)` taken before dropout, `device` (`cuda` or `cpu`), and `model_name` `roberta-base`
- `transcribe_audio(audio_path)` returns the ASR metadata above

`device` follows CUDA availability unless `MEMBER1_DEVICE` is set. Embeddings written for other members in `embeddings.npz` are float32 on CPU and row-aligned with utterance ids. DeBERTa embeddings go to a different file. MiniLM 384-d embeddings, if produced, go to `embeddings_minilm.npz`.

A missing checkpoint raises an error. The API does not return a dummy probability.

TF-IDF plus logistic regression, and a frozen `sentence-transformers/all-MiniLM-L6-v2` linear head, are baselines. They are logged separately. They are not the primary model, and their files are not copied over the RoBERTa checkpoint.

The implemented word+character TF-IDF baseline can be evaluated on MUStARD with:

```bash
PYTHONPATH=member1_text/src python -m member1_text.baselines
```

This runs five stratified folds and writes `outputs/eval/mustard__sarcasm/tfidf_baseline_cv.json`. It is a validation estimate, not an independent test score.

Reported numbers come only from `member1_text/outputs/eval/` after the matching command succeeds.

### Member 1 handoff

`member1_text/handoff/` contains the RoBERTa five-fold validation report, the TF-IDF baseline report, and the exported MUStARD embeddings (`ids`, `utterance`, `context`, `fused`, and `probability_sarcasm`). Embedding rows are float32 and aligned with `ids`. Both reports are cross-validation results; neither is an independent test result. The trained RoBERTa weight file is about 476 MB and is intentionally not included in the regular GitHub commit. Keep the deployment checkpoint at `outputs/checkpoints/mustard__sarcasm__roberta-base/` locally or share it separately with teammates who need direct inference.

## Tests

Stage 1 tests use the standard library and do not download models or corpora:

```bash
python3.11 -m unittest discover -s member1_text/tests -v
```

`test_version.py` checks that import does not exit, that Python 3.11 can import the package, and that `python -m member1_text` exits with the message above on another version. `test_scaffold.py` checks the tree, `config.yaml`, `requirements.txt`, `.gitignore`, the reporting rules, and that the repository root files are unchanged.
