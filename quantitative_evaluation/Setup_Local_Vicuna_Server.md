# Setup OpenAI Compatible FastChat API

* Install FastChat as mentioned [here](https://github.com/lm-sys/FastChat/tree/main). (For more details about the API, look [here](https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md).)

* Run the following commands to serve Vicuna-13B-v1.5 model

    CUDA_VISIBLE_DEVICES=4,5 python3 -m fastchat.serve.controller

    CUDA_VISIBLE_DEVICES=4,5 python3 -m fastchat.serve.model_worker --model-path lmsys/vicuna-13b-v1.5

    CUDA_VISIBLE_DEVICES=4,5 python3 -m fastchat.serve.openai_api_server --host localhost --port 8000


    export OPENAI_API_BASE=http://localhost:8000/v1
    export OPENAI_API_KEY=EMPTY
