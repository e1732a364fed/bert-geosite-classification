# geosite-gfw

本项目是一个机器学习训练http内容二分类的pytorch 项目。

先利用 geosite 的网站列表下载数据，然后用 bert 微调的方式训练模型。

提供了 用于 识别的 api。

模型下载地址在
https://huggingface.co/e1732a364fed/geosite-gfw/tree/main

下载两个zip文件后解压到 项目中即可

```sh
curl -LO "https://huggingface.co/e1732a364fed/geosite-gfw/resolve/main/bert_geosite_by_body.zip?download=true"
curl -LO "https://huggingface.co/e1732a364fed/geosite-gfw/resolve/main/bert_geosite_by_head.zip?download=true"

tar -xf bert_geosite_by_body.zip
tar -xf bert_geosite_by_head.zip
```

下载好模型后就可直接跳到下面第三步进行预测了

本项目已在 [ruci](https://github.com/e1732a364fed/ruci) 代理项目中使用 :[geosite_gfw](https://e1732a364fed.github.io/ruci/lua/route_config.html#geosite_gfw)

# Steps

0. install requirements

```
pip install transformers numpy scikit-learn flask requests
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

download `bert-base-multilingual-cased` from huggingface, store the files in ./bert/ folder

if you are not using nvidia gpu, you may obmit the  --index-url parameter.


## 1. pull geosite response with `python pull.py`

you can set the read list by `python pull.py -l proxy-list.txt`

the project contains 2 list from 
https://github.com/Loyalsoldier/v2ray-rules-dat

but maybe dated. You can use your own list file.

## 2. train model with

```
python classify.py --mode=train_head
python classify.py --mode=train_body
```

it will generate the trained model file.

you can set the ok and ban dir by --ok_dir and --ban_dir

## 3. predict with

```
python classify.py --mode predict_head --text "Your input text here"
python classify.py --mode predict_body --text "Your input text here"
```

You can download pretrained model files instead of training own your own.


## 4. serve api with

python classify.py --mode serve_api --port 5134

## 5. request

### predict by passing the data

```bash
curl -X POST http://localhost:5134/predict \
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

## check by passing the domain

```bash
curl -X POST http://localhost:5134/check \
    -H "Content-Type: application/json" \
    -d '{"domain": "www.baidu.com"}'
```

1. for more arguments and options, see the source code

