# geosite-gfw

0. install requirements

pip install transformers numpy scikit-learn flask requests
pip install torch --index-url https://download.pytorch.org/whl/cu124

download bert-base-multilingual-cased from huggingface, store the files in ./bert/ folder


1. pull geosite response with python pull.py
2. train model with

```
python classify.py --mode=train_head
python classify.py --mode=train_body
```

3. predict with
```
python classify.py --mode predict_head --text "Your input text here"
python classify.py --mode predict_body --text "Your input text here"
```

4. serve api with
python classify.py --mode serve_api --port 5000

5. request with

```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "your website http response head", "model_type": "head"}'
```
- response:
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



6. for more arguments and options, see the source code
