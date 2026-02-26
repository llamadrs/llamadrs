# LlaMADRS

Code for the paper:

**LlaMADRS: Evaluating Open-Source LLMs on Real Clinical Interviews — To Reason or Not to Reason?**

This repo contains scripts for preparing CAMI interview audio/transcripts and running LLM-based diarization, MADRS segmentation, and model evaluation.

> Note: CAMI audio and raw transcripts are sensitive and are not distributed with this repository.

---

## CAMI input data structure

A single CAMI session directory typically looks like:



## Prompts
The `prompts` directory contains various prompt files used for MADRS assessment and segmentation. The directory is organized into several subdirectories, each representing a specific prompting strategy or variant:

1. `full`: Contains the complete set of prompts for each MADRS item, including descriptions, demonstrations, and guidelines for assessment.

2. `no_dem`: Prompts in this directory exclude the demonstrative examples.

3. `no_desc`: This directory contains prompts without the descriptive content or explanations.

5. `raw`: This directory contains the initial or unprocessed versions of the prompts.

The `madrs_questions_prompt.txt` file in the root of the `prompts` directory is used by the `llm_segment.py` script to map utterances to MADRS domains. It contains the questions or items that are relevant for each domain of the MADRS scale.