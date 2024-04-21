# test run llama3 with huggingface
- os windows
- Python 3.12.3 

## install
```
pip install huggingface_hub
pip install transformers==4.40.0
pip install accelerate
pip install bitsandbytes
```

## token txt
make "hf_token.txt" and paste token.

## download model
```
py .\download_model.py
```

## run
```
py .\llama3.py
```

```
...

WARNING:root:Some parameters are on the meta device device because they were offloaded to the cpu and disk.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:128001 for open-end generation.
まどか☆マギカのキャラクターたちはみなかわいいですが、各人には個人的な好みがあると考えられます。ただし、人気投票などをみると、HomuraとMadokaが一番かわいいと評判されています。特にHomuraのツンデレなキャラクター性とMadokaの純真な
性格がファンの人気を博しています。
===
1846.414874792099
```

## ref
https://huggingface.co/welcome  
https://note.com/npaka/n/n73b0786f48e9  
https://zenn.dev/timoneko/articles/764310d5b8bdd9  
https://qiita.com/DeepTama/items/b17ac21010424de38c5f