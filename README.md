# LlaMADRS: A Systematic Evaluation of Large Language Models for Clinical Depression Assessment in Psychiatric Interviews

This repository contains the code and resources for the paper "LlaMADRS: A Systematic Evaluation of Large Language Models for Clinical Depression Assessment in Psychiatric Interviews".

## CAMI Dataset Poster and links
The CAMI dataset is a unique resource that includes real patient-clinician interactions and their corresponding expert ratings for depression severity using the Montgomery-Åsberg Depression Rating Scale (MADRS). The dataset was created by researchers at Harvard Medical School and McLean Hospital.

- Project: [CAMI dataset](https://bakerlab.mclean.harvard.edu/context-adaptive-multimodal-informatics-for-psychiatric-discharge-planning/). 
- NIH RePORTER link: [NIH RePORTER](https://reporter.nih.gov/search/a9XdMHOQ20-T4NAcPOFU_Q/project-details/10167040).
- The Poster is in the [./CAMI_Poster.pdf](https://github.com/llamadrs/llamadrs/blob/main/CAMI_Poster.pdf) file.

## Code Overview

The repository contains the following key scripts:

1. `whisperx_transcription.sh`: Takes care of speech-to-text transcription using the WhisperX model. It processes audio files and generates transcripts in VTT format.

2. `llm_diarize.py`: Handles speaker diarization using the Qwen 2.5 - 72B model. It processes the transcripts and assigns speaker roles (Practitionr or Patient) to each utterance.

3. `llm_segment.py`: Maps utterances to MADRS domains, implementing the segmentation module. It analyzes the content of each utterance and classifies it into the relevant MADRS item.

4. `madrs_predict_holistic.py`: Performs MADRS scoring using different prompting strategies. It takes the full transcript (with diarization) and applies various prompting strategies to assess the depression severity based on the MADRS scale. It generates structured outputs including scores, explanations, and key utterances for each MADRS item.

5. `madrs_predict_segmented.py`: Similar to `madrs_predict_holistic.py`, but it operates on segmented utterances. It uses the output from the segmentation module to score the relevant segment of the conversation for each MADRS item.

6. `eva

These scripts form the core components of the LlaMADRS pipeline, handling preprocessing, assessment, and segmentation tasks as described in the paper.

## Prompts
The `prompts` directory contains various prompt files used for MADRS assessment and segmentation. The directory is organized into several subdirectories, each representing a specific prompting strategy or variant:

1. `full`: Contains the complete set of prompts for each MADRS item, including descriptions, demonstrations, and guidelines for assessment.

2. `no_dem`: Prompts in this directory exclude the demonstrative examples or illustrations.

3. `no_desc`: This directory contains prompts without the descriptive content or explanations.

4. `no_disc`: Prompts in this directory exclude any discursive or additional contextual information.

5. `raw`: This directory contains the initial or unprocessed versions of the prompts.

The `madrs_questions_prompt.txt` file in the root of the `prompts` directory is used by the `llm_segment.py` script to map utterances to MADRS domains. It contains the questions or items that are relevant for each domain of the MADRS scale.