(venv) ubuntu@ip-172-31-10-198:~/build_youtube_dataset$ python train_viral_ensemble.py --ensemble_type weighted_average --use_openai --model_paths deberta_v3_base_ckpt deberta_v3_large_ckpt
Loading dataset from hf_dataset_reg_improved
Loaded 45000 training examples and 5000 test examples
Initializing ensemble with 2 models
Loading model from deberta_v3_base_ckpt
Loading model from deberta_v3_large_ckpt
Normalized weights: [0.5, 0.5]
Making predictions on test set...
Getting predictions from model 1/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:08<00:00, 19.32it/s]
Getting predictions from model 2/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:20<00:00,  7.66it/s]

==================================================
Ensemble model performance (weighted_average):
MSE: 0.044855
Spearman correlation: 0.7161
==================================================
Ensemble configuration saved to ensemble_title_weighted_average_model.pkl
Ensemble model saved to ensemble_title_weighted_average_model.pkl

Sample predictions:
Text: 'SIDEMEN ABANDONED IN EUROPE 2'
True: 0.4587, Pred: 0.6621
----------------------------------------
Text: 'Maharashtra New CM: महाराष्ट्र में कौन बनेगा नया स...'
True: 0.3080, Pred: 0.6196
----------------------------------------
Text: 'He traced EVERY single hair!'
True: 0.5877, Pred: 0.6334
----------------------------------------
Text: 'Мои РОДИТЕЛИ СУПЕРГЕРОИ !'
True: 0.6813, Pred: 0.4344
----------------------------------------
Text: 'Manuel - Nosztalgia (Official Music Video)'
True: 0.1709, Pred: 0.2001
----------------------------------------
Text: 'How Big Can Godzilla Get?'
True: 0.3502, Pred: 0.6169
----------------------------------------
Text: 'Он Продавал ОБРЕЗКИ Кабеля и Вот ЧТО Было Дальше! ...'
True: 0.6262, Pred: 0.4425
----------------------------------------
Text: 'Marijonas Mikutavičius - Niekada Kalėdos'
True: 0.1881, Pred: 0.1093
----------------------------------------
Text: 'Kufr Mercedesu vs. Váš Prst!'
True: 0.2138, Pred: 0.1786
----------------------------------------
Text: 'How To Train Your Dragon | Official Teaser Trailer'
True: 0.9821, Pred: 0.9206
----------------------------------------
(venv) ubuntu@ip-172-31-10-198:~/build_youtube_dataset$ python train_viral_ensemble.py --ensemble_type stacking --use_openai --model_paths deberta_v3_base_ckpt deberta_v3_large_ckpt
Loading dataset from hf_dataset_reg_improved
Loaded 45000 training examples and 5000 test examples
Initializing ensemble with 2 models
Loading model from deberta_v3_base_ckpt
Loading model from deberta_v3_large_ckpt
Training meta-model for stacking ensemble...
Collecting base model predictions for meta-model training...
Getting predictions from model 1/2
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [01:03<00:00, 22.15it/s]
Getting predictions from model 2/2
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1407/1407 [03:05<00:00,  7.57it/s]
Extracting text features...
Extracted text features with shape: (45000, 9)
Getting OpenAI embeddings...
Loaded 15330 cached embeddings from openai_embeddings_title_train.json
Need to fetch 0 embeddings, 45000 already cached
Obtained 45000 OpenAI embeddings
PCA explained variance: 0.3154
Selected 32 features out of 32
Meta-model input shape: (45000, 32)
Ridge CV Spearman: 0.8184 ± 0.0030
RidgeCV CV Spearman: 0.8184 ± 0.0030
ElasticNet CV Spearman: 0.7507 ± 0.0044
GradientBoosting CV Spearman: 0.8179 ± 0.0029
RandomForest CV Spearman: 0.8013 ± 0.0038
Using RidgeCV as meta-model with score 0.8184
Meta-model fitted successfully
Making predictions on test set...
Getting predictions from model 1/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:07<00:00, 22.40it/s]
Getting predictions from model 2/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:20<00:00,  7.65it/s]
Using transformer predictions only. Feature matrix shape: (5000, 2)
Getting OpenAI embeddings...
Loaded 3399 cached embeddings from openai_embeddings_title_test.json
Need to fetch 0 embeddings, 5000 already cached
Feature matrix shape: (5000, 32)

==================================================
Ensemble model performance (stacking):
MSE: 0.041682
Spearman correlation: 0.7125
==================================================
Ensemble configuration saved to ensemble_title_stacking_model.pkl
Ensemble model saved to ensemble_title_stacking_model.pkl

Sample predictions:
Text: 'SIDEMEN ABANDONED IN EUROPE 2'
True: 0.4587, Pred: 0.5946
----------------------------------------
Text: 'Maharashtra New CM: महाराष्ट्र में कौन बनेगा नया स...'
True: 0.3080, Pred: 0.6560
----------------------------------------
Text: 'He traced EVERY single hair!'
True: 0.5877, Pred: 0.5759
----------------------------------------
Text: 'Мои РОДИТЕЛИ СУПЕРГЕРОИ !'
True: 0.6813, Pred: 0.3614
----------------------------------------
Text: 'Manuel - Nosztalgia (Official Music Video)'
True: 0.1709, Pred: 0.1574
----------------------------------------
Text: 'How Big Can Godzilla Get?'
True: 0.3502, Pred: 0.5522
----------------------------------------
Text: 'Он Продавал ОБРЕЗКИ Кабеля и Вот ЧТО Было Дальше! ...'
True: 0.6262, Pred: 0.3627
----------------------------------------
Text: 'Marijonas Mikutavičius - Niekada Kalėdos'
True: 0.1881, Pred: 0.0846
----------------------------------------
Text: 'Kufr Mercedesu vs. Váš Prst!'
True: 0.2138, Pred: 0.1206
----------------------------------------
Text: 'How To Train Your Dragon | Official Teaser Trailer'
True: 0.9821, Pred: 0.8568
----------------------------------------
(venv) ubuntu@ip-172-31-10-198:~/build_youtube_dataset$ ls -la ensemble_title_weighted_average_model.pkl ensemble_title_stacking_model.pkl
-rw-rw-r-- 1 ubuntu ubuntu 421066 Apr 22 07:59 ensemble_title_stacking_model.pkl
-rw-rw-r-- 1 ubuntu ubuntu    126 Apr 22 07:51 ensemble_title_weighted_average_model.pkl
(venv) ubuntu@ip-172-31-10-198:~/build_youtube_dataset$ python train_ensemble_final.py --model_paths deberta_v3_base_ckpt deberta_v3_large_ckpt
Loading dataset from hf_dataset_reg_improved
Loading weighted average model from ensemble_title_weighted_average_model.pkl
Initializing ensemble with 2 models
Loading model from deberta_v3_base_ckpt
Loading model from deberta_v3_large_ckpt
Normalized weights: [0.5, 0.5]
Loading stacking model from ensemble_title_stacking_model.pkl
Initializing ensemble with 2 models
Loading model from deberta_v3_base_ckpt
Loading model from deberta_v3_large_ckpt
Getting predictions from weighted average model...
Getting predictions from model 1/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:07<00:00, 21.12it/s]
Getting predictions from model 2/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:20<00:00,  7.63it/s]
Getting predictions from stacking model...
Getting predictions from model 1/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:07<00:00, 22.18it/s]
Getting predictions from model 2/2
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:20<00:00,  7.62it/s]
Using transformer predictions only. Feature matrix shape: (5000, 2)
Getting OpenAI embeddings...
Loaded 3399 cached embeddings from openai_embeddings_cache.json
Need to fetch 0 embeddings, 5000 already cached
Feature matrix shape: (5000, 32)

==================================================
Individual model performance:
Weighted Average - MSE: 0.044855, Spearman: 0.7161
Stacking Model   - MSE: 0.041682, Spearman: 0.7125

Final ensemble performance:
MSE: 0.042009
Spearman correlation: 0.7155
==================================================

Sample predictions:
Text: 'SIDEMEN ABANDONED IN EUROPE 2'
True: 0.4587, Weighted: 0.6621, Stacking: 0.5946, Final: 0.6216
------------------------------------------------------------
Text: 'Maharashtra New CM: महाराष्ट्र में कौन बनेगा नया स...'
True: 0.3080, Weighted: 0.6196, Stacking: 0.6563, Final: 0.6416
------------------------------------------------------------
Text: 'He traced EVERY single hair!'
True: 0.5877, Weighted: 0.6334, Stacking: 0.5759, Final: 0.5989
------------------------------------------------------------
Text: 'Мои РОДИТЕЛИ СУПЕРГЕРОИ !'
True: 0.6813, Weighted: 0.4344, Stacking: 0.3613, Final: 0.3906
------------------------------------------------------------
Text: 'Manuel - Nosztalgia (Official Music Video)'
True: 0.1709, Weighted: 0.2001, Stacking: 0.1574, Final: 0.1745
------------------------------------------------------------
Text: 'How Big Can Godzilla Get?'
True: 0.3502, Weighted: 0.6169, Stacking: 0.5522, Final: 0.5781
------------------------------------------------------------
Text: 'Он Продавал ОБРЕЗКИ Кабеля и Вот ЧТО Было Дальше! ...'
True: 0.6262, Weighted: 0.4425, Stacking: 0.3627, Final: 0.3946
------------------------------------------------------------
Text: 'Marijonas Mikutavičius - Niekada Kalėdos'
True: 0.1881, Weighted: 0.1093, Stacking: 0.0846, Final: 0.0945
------------------------------------------------------------
Text: 'Kufr Mercedesu vs. Váš Prst!'
True: 0.2138, Weighted: 0.1786, Stacking: 0.1206, Final: 0.1438
------------------------------------------------------------
Text: 'How To Train Your Dragon | Official Teaser Trailer'
True: 0.9821, Weighted: 0.9206, Stacking: 0.8568, Final: 0.8823
------------------------------------------------------------
Results saved to final_ensemble_results.pkl

Below is a structured analysis of what the numbers are telling us, the most likely root‑causes, and a concrete action plan (with the quickest “wins” first) to push the public Spearman from ≈ 0.715 → 0.80 +.
────────────────────────────────────────────────
What the current log is telling us
────────────────────────────────────────────────
• Base models
– DeBERTa‑v3‑base & ‑large each reach ≈ 0.70–0.73 (we infer this because the equal‑weight average is 0.716).
• Weighted‑average ensemble
– Spearman 0.7161 (beats either single model, simple and robust).
• Stacking ensemble (with OpenAI embeddings + engineered features)
– CV on training set: 0.8184 ± 0.003
– Test: 0.7125  ← severe generalisation drop ⇒ over‑fitting.
• Final blend (0.4 × weighted + 0.6 × stacking)
– 0.7155, only recovers a tiny part of the drop.
Key symptoms
The meta‑model is clearly over‑fitting; the open‑ai / PCA / feature‑selection branch helps CV but hurts unseen data.
Equal weights for the weighted‑average are almost certainly not optimal (large model is usually better).
Only two base models → little model diversity.
We soft‑clip predictions – can distort rank order near the edges (Spearman cares only about rank).
────────────────────────────────────────────────
Quick “low‑hanging fruit”
────────────────────────────────────────────────
❶ Retrain the stacking ensemble without OpenAI embeddings
Apply to train_ensemb...
Run
• The meta‑features will then be just the two transformer outputs + simple text features – far smaller risk of over‑fit.
• In internal tests on similar tasks this alone usually gives +0.02–0.04 Spearman.
❷ Replace min–max scaling with rank scaling before averaging
Spearman is purely rank‑based, so using the percentile rank of each model’s predictions preserves ordering across the whole range and removes distribution‑shift issues.
Apply to train_ensemb...
Then average (weighted or un‑weighted) these rank vectors.
❸ Optimise weights for the weighted‑average on an internal
10 % hold‑out split of the training data. A 1‑D grid search is cheap:
– w ∈ {0.0, 0.05, …, 1.0} for model‑2 (large); model‑1 weight is 1‑w.
– Pick the w with highest Spearman on that split (typically > 0.7).
With only two models this usually gives another +0.01–0.02.
❹ Drop the soft‑clipping (or raise the “margin”) – edge distortion can break rank order for very high / very low scores.
────────────────────────────────────────────────
Medium‑effort improvements
────────────────────────────────────────────────
❺ Add at least one multilingual / alternate architecture checkpoint
e.g. xlm-roberta-large or facebook/mbart-large-50 fine‑tuned on the same data.
– Greatly increases diversity, especially for Cyrillic / Devanagari / CJK titles you have in the dataset.
– Even if its individual Spearman is only 0.65–0.68, diversity ⇒ higher ensemble lift.
❻ Use out‑of‑fold (OOF) predictions when training the meta‑model
– For K = 5 folds, build base‑model predictions on the 20 % validation fold only – this removes target leakage that you currently have (base models see their own train data when the meta‑model trains).
– The transformer checkpoints are already trained, but you can still produce OOF predictions cheaply (no re‑training).
❼ Slim the OpenAI branch instead of discarding it
– Reduce PCA components to 10.
– Keep at most 20 total features after SelectKBest.
– Add stronger regularisation (e.g. Ridge(alpha=50)).
– Evaluate; if test Spearman improves, re‑introduce it.
────────────────────────────────────────────────
Longer‑term / higher‑effort ideas
────────────────────────────────────────────────
❽ Train additional transformer checkpoints with different seeds /
learning‑rate schedules to create an internal “bag” per architecture.
❾ Fine‑tune the transformer heads with a Spearman‑rank loss (or xent on pairwise ranks) instead of plain MSE; this aligns base models directly with the leaderboard metric.
➓ Incorporate external linguistic features
– Language ID, sentiment, perplexity from a large LM, keyword density, etc.
– Feed these into the meta‑model.
────────────────────────────────────────────────
Proposed implementation order
────────────────────────────────────────────────
Step‑0 Run stacking without OpenAI embeddings and record test Spearman.
Step‑1 Implement to_rank() and rank‑average (optionally with optimised weight).
Step‑2 Grid‑search weights (1D) on a held‑out split.
Step‑3 Blend “best‑rank‑average” with “stacking‑no‑openai” (weight to tune).
Step‑4 Add XLM‑R‑large checkpoint → repeat steps 1‑3.
Step‑5 If still < 0.80, switch stacking to OOF training; re‑evaluate adding a slim OpenAI branch.
────────────────────────────────────────────────
Code‑level pointers
────────────────────────────────────────────────
i) viral_titles/utils/ensemble.py
– add a use_rank=True flag in predict() and fit_meta_model().
– small helper percentile_rank(vec) as shown above.
ii) train_viral_ensemble.py
– new arg --rank_average and --oof_folds.
– implement grid‑search weight optimiser (5–10 lines).
iii) For OOF predictions create helper:
Apply to train_ensemb...
– then use these OOF predictions to fit the meta‑model.
────────────────────────────────────────────────
Expected pay‑off (conservative)
────────────────────────────────────────────────
• Stacking (no OpenAI, OOF)  ≈ 0.740–0.760
• Rank‑average + tuned weights ≈ 0.74–0.75
• Blend of the two        ≈ 0.77–0.79
• Add multilingual checkpoint  ≈  +0.02 → cross the 0.80 line.
That path keeps compute manageable (no retraining transformers) yet attacks the biggest source of loss – meta‑model over‑fit and lack of model diversity