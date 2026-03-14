# swiss_AI_discourse



## 1. GPU Server usage

- Follow instructions on the guide sent via email for initial setup.

- Uploading files (input commands from local terminal)

    ```
    scp -r <Key Directory> "<local folder path>" <user name>@<server name>.cl.uzh.ch:/home/<user name>/<folder_name>
    ```

- Synchronizing files (input commands from local terminal)

    - Recommend to use Linux terminal.

    - server to local
        ```
        rsync -asv --ignore-existing -e "<Key Directory>" <user name>@<server name>.cl.uzh.ch:/home/<user name>/<folder name> <local folder path>
        ```

    - local to server
        ```
        rsync -asv --ignore-existing -e "<Key Directory>" <local folder path> <user name>@<server name>.cl.uzh.ch:/home/<user name>/<folder name>
        ```

- Running multiple sessions
    - Create a new session
    ```
    tmux new -s <session name>
    ```
    - access existing session
    ```
    tmux attach -t <session name>
    ```
    - Return to the main session
    ```
    Ctrl + b, d
    ```
    - Closing session
    ```
    exit
    ```

## 2. Project Structure

- data: raw data and processed data (down load from query results)
    - file names: de.tsv, fr.tsv, de_edu.tsv,...

- topic_extraction: code for topic extraction
    
## 3. LLM Module

Repository for LLM classification and summarization module.

1. Run a vllm server for initiating LLM

    Create a new session for running the vllm server, and run the following command in the terminal. This will start a vllm server on port 8000.
    
    **Please close the entrypoint after use with `ctrl + c`! This occupies the port and resources until it is closed.**
    ```
    python -m vllm.entrypoints.openai.api_server \
        --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
        --served-model-name llama \
        --enable-lora \
        --max-model-len 8192 \
        --max-num-seqs 128 \
        --trust-remote-code \
        --port 8000 \
        --distributed-executor-backend mp
    ```

2. Run a classification pipeline

    Create a new session for running the classification pipeline, and run the following command in the terminal. This will run the classification pipeline using the vllm server on port 8000.
    
    ```
    python -m llm_module.run

    python -m llm_module.run --url 8000 --model_name llama --input_file ./data/politics/cleaned_trial_data.csv --output_file run_8000.json --categories "Asylum,Integration,Economy,Politics,Security" --mode all
    
    python -m llm_module.run --categories "[\"Asylum\",\"Integration\",\"Economy\"]" --mode classify
    
    python -m llm_module.run --mode summarize --output_file summaries.json
    
    python -m llm_module.run --mode verify --output_file verify.json
    ```

    Please tune prompts or command on your own for better performance. The current configuration is set to classify documents into 5 categories: "Asylum", "Integration", "Economy", "Politics", "Security". You can change the categories and system prompt in the `Config` object in `run.py`.