from huggingface_hub import login, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time

TOKEN_TXT = "hf_token.txt"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_PATH ="model"
HF_HOME = r"D:/Users/Kazu/Documents/Project/llama3/.cache/huggingface"

def hf_login():
    token = ''
    with open(TOKEN_TXT, 'r') as f:
        token = f.read()
    login(token=token)

def download_model():
    download_path = snapshot_download(repo_id=MODEL_ID,
                                  local_dir=LOCAL_PATH,
                                  local_dir_use_symlinks=False)
    print(download_path)


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

def get_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        offload_folder=LOCAL_PATH
    )

def main():
    # os.environ['HF_HOME'] = HF_HOME

    hf_login()

    # download_model() # 初回のみ

    tokenizer = get_tokenizer()
    model = get_model()

    # プロンプトの準備
    chat = [
        { "role": "system", "content": "あなたは日本語で回答するAIアシスタントです。" },
        { "role": "user", "content": "まどか☆マギカでは誰が一番かわいい?" },
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    start = time.time()
    # 推論の実行
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            temperature=0.6,
            top_p=0.9,        
            max_new_tokens=256,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
    end = time.time()

    print(output)

    print("===")
    print(end - start)

if __name__ == '__main__':
    main()