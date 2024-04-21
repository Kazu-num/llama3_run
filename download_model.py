from huggingface_hub import login, snapshot_download

TOKEN_TXT = "hf_token.txt"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_PATH ="model"

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

def main():
    hf_login()
    download_model()

if __name__ == '__main__':
    main()