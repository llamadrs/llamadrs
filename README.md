# LlaMADRS: Evaluating Open-Source LLMs on Real Clinical Interviews—To Reason or Not to Reason?

This repository contains the code and resources for the paper "LlaMADRS: Evaluating Open-Source LLMs on Real Clinical Interviews—To Reason or Not to Reason?".

## CAMI Dataset Poster and links
The CAMI dataset is a unique resource that includes real patient-clinician interactions and their corresponding expert ratings for depression severity using the Montgomery-Åsberg Depression Rating Scale (MADRS). The dataset was created by researchers at Harvard Medical School and McLean Hospital.

- Project: [CAMI dataset](https://bakerlab.mclean.harvard.edu/context-adaptive-multimodal-informatics-for-psychiatric-discharge-planning/). 
- NIH RePORTER link: [NIH RePORTER](https://reporter.nih.gov/search/a9XdMHOQ20-T4NAcPOFU_Q/project-details/10167040).
- The Poster is in the [./CAMI_Poster.pdf](https://github.com/llamadrs/llamadrs/blob/main/CAMI_Poster.pdf) file.

## Code Overview


## Prompts
The `prompts` directory contains various prompt files used for MADRS assessment and segmentation. The directory is organized into several subdirectories, each representing a specific prompting strategy or variant:

1. `full`: Contains the complete set of prompts for each MADRS item, including descriptions, demonstrations, and guidelines for assessment.

2. `no_dem`: Prompts in this directory exclude the demonstrative examples.

3. `no_desc`: This directory contains prompts without the descriptive content or explanations.

5. `raw`: This directory contains the initial or unprocessed versions of the prompts.

The `madrs_questions_prompt.txt` file in the root of the `prompts` directory is used by the `llm_segment.py` script to map utterances to MADRS domains. It contains the questions or items that are relevant for each domain of the MADRS scale.