# Bert-Geosite-Classification

本项目是一个机器学习训练http内容二分类(Binary Classification)的pytorch 项目。

先利用 geosite 的网站列表下载数据，然后用 bert 微调的方式训练模型。

提供了 用于 识别的 api。

git clone 下面两个模型

https://huggingface.co/e1732a364fed/bert-geosite-classification-head-v1/tree/main

https://huggingface.co/e1732a364fed/bert-geosite-classification-body-v1/tree/main

然后对文件夹分别改名为 bert_geosite_by_body 和 bert_geosite_by_head。

下载好模型后就可直接跳到下面第三步进行预测了

本项目已在 [ruci](https://github.com/e1732a364fed/ruci) 代理项目中使用 :[geosite_gfw](https://e1732a364fed.github.io/ruci/lua/route_config.html#geosite_gfw)

# Steps

0. install requirements

```sh
pip install transformers numpy scikit-learn flask requests
pip install "requests[socks]"
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

download `bert-base-multilingual-cased` from huggingface, store the files in ./bert/ folder

if you are not using nvidia gpu, you may obmit the  --index-url parameter.

或者直接使用

```sh
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

(生成 requirements.txt 的命令是 uv pip freeze > requirements.txt)

如果您想用 venv 而不是uv, 就是运行如下命令

```sh
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```


## 1. pull geosite responses data with `python pull.py`

or you can set the geosite list by `python pull.py -l proxy-list.txt`

the project contains 2 geosite list from 
https://github.com/Loyalsoldier/v2ray-rules-dat

but maybe dated. You can use your own list file.

会生成 {list_name}_out 文件夹, 里面为每一个网站的响应


## 2. train model with

```sh
python classify.py --mode=train_head
python classify.py --mode=train_body
```

it will generate the trained model file.

you can set the ok and ban dir by --ok_dir and --ban_dir

You can download pretrained model files instead of training own your own.


## 3. predict with

```sh
python classify.py --mode predict_head --text "Your input text here"
python classify.py --mode predict_body --text "Your input text here"
```


for example, 
```sh
# this well return Prediction: ban
python classify.py --mode predict_body --text "<body>google</body>"

# this well return Prediction: ban
python classify.py --mode predict_body --text "<body>porn</body>"

# this well return Prediction: ok
python classify.py --mode predict_body --text "<body>baidu</body>"
```




## 4. serve api with

```sh
python classify.py --mode serve_api --port 5134
```

mac上测试, 内存占用325.8MB



## 5. request

### predict by passing the data

```bash
curl -X POST http://localhost:5134/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "<body>your website http response body</body>", "model_name": "body"}'
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

with proxy:
```bash
curl -X POST http://localhost:5134/check \
    -H "Content-Type: application/json" \
    -d '{"domain": "www.google.com", "socks5_proxy": "127.0.0.1:10800"}'
```



1. for more arguments and options, see the source code

# benchmark

there's a benchmark.py that benches cpu and mps. 
On macOS, mps is way faster than cpu.
run `python benchmark.py` to see how fast it is on your mac.