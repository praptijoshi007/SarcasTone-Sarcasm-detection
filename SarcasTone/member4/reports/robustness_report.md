# SarcasTone Multimodal Robustness & Stress-Testing Report

**Evaluation Date**: 2026-11-23T21:40:00Z  
**Overall Robustness Index**: **92.57%**  
**All Graceful Fallbacks Maintained**: True  

## Empirical Benchmark Results

| Scenario / Perturbation | Description | Macro $F_1$ | Accuracy | Calibration Error (ECE) | Degradation | Graceful Fallback |
|---|---|---|---|---|---|:---:|
| `clean_baseline` | Pristine studio audio and verified ground-truth transcript with context | **0.7425** | 0.7460 | 0.128 | Baseline | ✅ |
| `background_noise_10db` | Simulated environmental acoustic noise (SNR 10 dB) | **0.6994** | 0.7050 | 0.154 | 5.8% | ✅ |
| `reduced_audio_quality_clipping` | Signal degradation: 20% hard clipping and low-pass bandwidth constriction | **0.6794** | 0.6863 | 0.172 | 8.5% | ✅ |
| `missing_audio_text_only` | Complete acoustic dropout: evaluating fallback to unimodal text encoding | **0.6980** | 0.7101 | 0.198 | 6.0% | ✅ |
| `missing_transcript_asr_only` | Omission of ground-truth transcript: audio transcribed via upstream ASR | **0.6905** | 0.6953 | 0.165 | 7.0% | ✅ |
| `asr_transcription_errors_wer20` | Simulated speech recognition noise (20% Word Error Rate) | **0.6645** | 0.6721 | 0.188 | 10.5% | ✅ |
| `context_removal_ablation` | Ablation of preceding conversational turns (isolated target utterance) | **0.6868** | 0.6938 | 0.145 | 7.5% | ✅ |
| `speaker_independent_testing` | Strict character-disjoint cross-validation (evaluating on unseen speakers) | **0.6930** | 0.7015 | 0.162 | 6.7% | ✅ |

## Key Findings
1. **Quality-Aware Gating Mitigation**: When acoustic noise (SNR 10 dB) is injected, the confidence gate automatically reduces audio contribution by ~18%, preventing error cascade.
2. **Graceful Text-Only Degradation**: Complete loss of audio reduces Macro $F_1$ from 0.7425 to 0.6980 (matching Member 1 unimodal RoBERTa ceiling) without catastrophic crash.
3. **Speaker-Independent Generalization**: Under disjoint speaker evaluation, SarcasTone maintains a 0.6930 Macro $F_1$, confirming that cross-attention generalizes beyond vocal identity memorization.