# geosite-gfw

0. install requirements

```
pip install transformers numpy scikit-learn flask requests
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

download `bert-base-multilingual-cased` from huggingface, store the files in ./bert/ folder


1. pull geosite response with `python pull.py`

you can set the read list by `python pull.py -l proxy-list.txt`

the project contains 2 list from 
https://github.com/Loyalsoldier/v2ray-rules-dat

but maybe dated. You can use your own list file.

2. train model with

```
python classify.py --mode=train_head
python classify.py --mode=train_body
```

it will generate the trained model file.

you can set the ok and ban dir by --ok_dir and --ban_dir

3. predict with

```
python classify.py --mode predict_head --text "Your input text here"
python classify.py --mode predict_body --text "Your input text here"
```

You can download pretrained model files instead of training own your own.


4. serve api with

python classify.py --mode serve_api --port 5000

5. request with

```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "your website http response body", "model_name": "body"}'
```



response:

```json
{
  "result": "ok"
}
```
or
```json
{
  "result": "ban"
}
```

or
```bash
curl -X POST http://localhost:5000/check \
    -H "Content-Type: application/json" \
    -d '{"domain": "www.baidu.com"}'
```

6. for more arguments and options, see the source code
