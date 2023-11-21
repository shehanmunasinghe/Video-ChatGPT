# Prerequisites

* Setup https://github.com/hkchengrex/Tracking-Anything-with-DEVA
* Save or symlink all the tracker weights at `Video-ChatGPT/grounding_evaluation/weights`

# Quantitative Evaluation

## VidSTG Dataset

* Download the VidSTG dataset from [here](https://github.com/Guaranteer/VidSTG-Dataset).

### Preprocessing the dataset

    python grounding_evaluation/datasets/preproc_vidstg.py \
        --vidor_annotations_dir <vidor_annotations_dir> \
        --vidstg_annotations_dir <vidstg_annotations_dir>

### Running Evaluation

    cd Video-ChatGPT
    export PYTHONPATH="./:$PYTHONPATH"

* E.g. : Evaluating Video-ChatGPT

<!-- v1 -->
    python grounding_evaluation/eval_grounding.py \
        --model video_chatgpt \
        --model-name path/to/LLaVA-7B-Lightening-v1-1 \
        --projection_path path/to/video_chatgpt-7B.bin \
        --output_dir <your_output_directory> \
        --resolution 224 \
        --dataset vidstg \
        --vid_dir <vidstg_video_directory>  \
        --ann_dir <vidstg_annotation_directory>

* E.g. : Evaluating Ours(13B)

<!-- 13B -->
    python grounding_evaluation/eval_grounding.py \
        --model video_chatgpt \
        --model-name path/to/llava-v1.5-13b \
        --projection_path path/to/mm_projector.bin \
        --output_dir <your_output_directory> \
        --resolution 336 \
        --dataset vidstg \
        --vid_dir <vidstg_video_directory>  \
        --ann_dir <vidstg_annotation_directory>




## HCSTVG Dataset

* Download the HC-STVG dataset from [here](https://github.com/tzhhhh123/HC-STVG).
* Download the extracted question-answer pairs from [here](??).
<!-- TODO: Upload the QA pairs to onedrive and share -->

### Preprocessing the Dataset

    python grounding_evaluation/datasets/preproc_hcstvgv2.py \
        --video_dir <hcstvg_video_dir> \
        --ann_dir <hcstvg_annotations_dir>

### Running Evaluation

    cd Video-ChatGPT
    export PYTHONPATH="./:$PYTHONPATH"

* E.g. : Evaluating Video-ChatGPT

<!-- v1 -->
    python grounding_evaluation/eval_grounding.py \
        --model video_chatgpt \
        --model-name path/to/LLaVA-7B-Lightening-v1-1 \
        --projection_path path/to/video_chatgpt-7B.bin \
        --output_dir <your_output_directory> \
        --resolution 224 \
        --dataset hcstvg \
        --vid_dir <hcstvg_video_directory>  \
        --ann_dir <hcstvg_annotation_directory> \
        --hcstvg_qa_dir <hcstvg_qa_directory>

* E.g. : Evaluating Ours(13B)
<!-- 13B -->
    python grounding_evaluation/eval_grounding.py \
        --model video_chatgpt \
        --model-name path/to/llava-v1.5-13b \
        --projection_path path/to/mm_projector.bin \
        --output_dir <your_output_directory> \
        --resolution 336 \
        --dataset hcstvg \
        --vid_dir <hcstvg_video_directory>  \
        --ann_dir <hcstvg_annotation_directory> \
        --hcstvg_qa_dir <hcstvg_qa_directory>


# Qualitative Analysis

    cd Video-ChatGPT
    export PYTHONPATH="./:$PYTHONPATH"

* E.g. : Evaluating Video-ChatGPT
<!-- v1 -->
    python grounding_evaluation/gen_qualitative_results.py \
        --model-name path/to/LLaVA-7B-Lightening-v1-1 \
        --projection_path path/to/video_chatgpt-7B.bin \
        --input_video_path <input_video_path> \
        --output_video_path <output_video_path>


* E.g. : Evaluating Ours(13B)
<!-- v1.5-13b -->
    python grounding_evaluation/gen_qualitative_results.py \
        --model-name path/to/llava-v1.5-13b \
        --projection_path path/to/mm_projector.bin \
        --input_video_path <input_video_path> \
        --output_video_path <output_video_path>
