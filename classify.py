import os
import argparse
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    get_linear_schedule_with_warmup,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from flask import Flask, request, jsonify
import requests
import pickle

CACHE_FILE = "samples.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEAD_MAX_LEN = 512
BODY_MAX_LEN = 512
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
SEED = 42

BERT_DIR = './bert/' # bert-base-multilingual-cased
ROBERTA_DIR = './roberta/' # roberta-large
XLM_ROBERTA = './xlm_roberta' # xlm-roberta-large

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

model_classes = {
    'bert': (BertTokenizer, BertForSequenceClassification, BERT_DIR),
    'roberta': (RobertaTokenizer, RobertaForSequenceClassification, ROBERTA_DIR),
    'xlm': (AutoTokenizer, AutoModelForSequenceClassification, XLM_ROBERTA),
}

def extract_body_text_by_file(file_path, max_len=512):
    file_size = os.path.getsize(file_path)

    if file_size <= max_len:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()

    chunk_size = max_len // 3  # 三等分
    positions = [0, file_size // 2, file_size - chunk_size]  # 头、中、尾

    extracted_text = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for pos in positions:
            f.seek(pos)  # 定位到相应位置
            text_chunk = f.read(chunk_size)
            extracted_text.append(text_chunk.strip())

    return ''.join(extracted_text)

def extract_body_text_by_string(input_string, max_len=512):
    string_length = len(input_string)

    if string_length <= max_len:
        return input_string.strip()

    chunk_size = max_len // 3  # 三等分
    positions = [0, string_length // 2, string_length - chunk_size]  # 头、中、尾

    extracted_text = []

    for pos in positions:
        text_chunk = input_string[pos:pos + chunk_size]
        extracted_text.append(text_chunk.strip())

    return ''.join(extracted_text)

def collect_samples(ok_dir, ban_dir):
    """从目录中收集数据，优先从缓存文件 samples.pkl 读取"""
    
    # 1. 先尝试加载缓存数据
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                print("Loading samples from cache...")
                return pickle.load(f)
        except Exception as e:
            print(f"Failed to load cache: {e}")

    # 2. 重新收集数据
    print("Collecting samples from directories...")
    samples = []

    def process_directory(directory, label):
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} not found!")
            return

        for domain in os.listdir(directory):
            domain_path = os.path.join(directory, domain)
            head_path = os.path.join(domain_path, "head.txt")
            body_path = os.path.join(domain_path, "body.txt")

            if os.path.exists(head_path) and os.path.exists(body_path):
                try:
                    with open(head_path, 'r', encoding='utf-8', errors='ignore') as f:
                        head_text = f.read().strip()

                    body_text = extract_body_text_by_file(body_path, BODY_MAX_LEN)

                    samples.append({
                        "head": head_text,
                        "body": body_text,
                        "label": label
                    })
                except Exception as e:
                    print(f"Error reading files for domain {domain}: {e}")

    process_directory(ok_dir, label=0)
    process_directory(ban_dir, label=1)

    # 3. 序列化 samples 并缓存
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(samples, f)
            print("Samples cached successfully.")
    except Exception as e:
        print(f"Failed to save cache: {e}")

    return samples

# 自定义Dataset类
class TextDataset(Dataset):
    def __init__(self, samples, tokenizer, max_len, mode='head'):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mode = mode  # 'head' or 'body'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 读取对应模式的文本内容
        text = sample[self.mode]

        # Tokenize文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(sample['label'], dtype=torch.long)
        }

# 训练函数
def train_model(model, train_loader, val_loader, model_save_path):
    print("train_model")
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)
        val_acc, val_f1 = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            torch.save(model.state_dict(), model_save_path)
            best_f1 = val_f1
            print(f"New best model saved with F1: {best_f1:.4f}")

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].cpu().numpy()

            outputs = model(
                input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            predictions.extend(preds)
            true_labels.extend(labels)

    acc = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    return acc, f1

# 预测函数（统一用于 API 和命令行模式）
def predict_text(text, model, tokenizer, max_len):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        print("outputs", outputs)
        logits = outputs.logits.cpu().numpy()
        pred = np.argmax(logits, axis=1)[0]

    return "ok" if pred == 0 else "ban"


def train_model_pipeline(ok_dir, ban_dir, model_type, max_len, model_name):
    if model_type not in model_classes:
        raise ValueError("Invalid model_type. Use 'bert', 'roberta', or 'xlm'.")

    samples = collect_samples(ok_dir, ban_dir)
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, stratify=[s['label'] for s in samples], random_state=SEED
    )

    tokenizer_cls, model_cls, model_dir = model_classes[model_type]
    tokenizer = tokenizer_cls.from_pretrained(model_dir)
    model = model_cls.from_pretrained(model_dir, num_labels=2).to(DEVICE)

    train_dataset = TextDataset(train_samples, tokenizer, max_len, model_name)
    val_dataset = TextDataset(val_samples, tokenizer, max_len, model_name)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    train_model(model, train_loader, val_loader, f"best_{model_name}_model.bin")

    model_save_path = f"{model_type}_geosite_by_{model_name}" # 使用命令行参数中的 model_type
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)


def train_head_model(ok_dir, ban_dir, model_type):
    train_model_pipeline(ok_dir, ban_dir, model_type, HEAD_MAX_LEN, 'head')


def train_body_model(ok_dir, ban_dir, model_type):
    train_model_pipeline(ok_dir, ban_dir, model_type, BODY_MAX_LEN, 'body')


# it's in standard library
import sqlite3

# 初始化 SQLite 数据库
def init_db():
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS domain_predictions (
            domain TEXT PRIMARY KEY,
            head_prediction TEXT,
            body_prediction TEXT
        )
    """)
    conn.commit()
    conn.close()



def get_cache(domain):
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT head_prediction, body_prediction FROM domain_predictions WHERE domain=?", (domain,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"head_prediction": row[0], "body_prediction": row[1]}
    return None

def set_cache(domain, head_prediction, body_prediction):
    conn = sqlite3.connect("cache.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO domain_predictions (domain, head_prediction, body_prediction)
        VALUES (?, ?, ?)
        ON CONFLICT(domain) DO UPDATE SET
            head_prediction=excluded.head_prediction,
            body_prediction=excluded.body_prediction
    """, (domain, head_prediction, body_prediction))
    conn.commit()
    conn.close()

def start_api_service(ip, port, model_type, load_method):
    app = Flask(__name__)

    if model_type not in model_classes:
        raise ValueError("Invalid model_type. Use 'bert', 'roberta', or 'xlm'.")

    tokenizer_cls, model_cls, model_dir = model_classes[model_type]

    if load_method == 'state_dict':
        head_tokenizer = tokenizer_cls.from_pretrained(model_dir)
        body_tokenizer = tokenizer_cls.from_pretrained(model_dir)
        head_model = model_cls.from_pretrained(model_dir, num_labels=2).to(DEVICE)
        body_model = model_cls.from_pretrained(model_dir, num_labels=2).to(DEVICE)
    
        model_paths = {"head": "best_head_model.bin", "body": "best_body_model.bin"}
        head_model.load_state_dict(torch.load(model_paths["head"]))
        body_model.load_state_dict(torch.load(model_paths["body"]))
    elif load_method == 'pretrained':
        head_tokenizer = tokenizer_cls.from_pretrained(f"{model_type}_geosite_by_head")
        body_tokenizer = tokenizer_cls.from_pretrained(f"{model_type}_geosite_by_body")
        head_model = model_cls.from_pretrained(f"{model_type}_geosite_by_head").to(DEVICE)
        body_model = model_cls.from_pretrained(f"{model_type}_geosite_by_body").to(DEVICE)
    else:
        raise ValueError(f"Invalid load_method: {load_method}. Use 'state_dict' or 'pretrained'.")

    head_model.eval()
    body_model.eval()
    
    init_db()  # 确保数据库表存在

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.get_json()
        text = data.get('text', '')
        model_name = data.get('model_name', 'head')

        if model_name not in ['head', 'body']:
            return jsonify({'error': 'Invalid model_type. Use "head" or "body".'}), 400

        if model_name == "body":
            text = extract_body_text_by_string(text)

        model = head_model if model_name == 'head' else body_model
        tokenizer = head_tokenizer if model_name == 'head' else body_tokenizer

        max_len = HEAD_MAX_LEN if model_name == 'head' else BODY_MAX_LEN

        result = predict_text(text, model, tokenizer, max_len)
        return jsonify({'result': result})

    @app.route('/check', methods=['POST'])
    def check():
        data = request.json
        domain = data.get('domain')
        socks5_proxy = data.get('socks5_proxy')
        only_proxy = data.get('only_proxy', False)

        if not domain:
            return jsonify({'error': 'Domain is required'}), 400

        # 1. 查询缓存
        cached_result = get_cache(domain)
        if cached_result:
            return jsonify({
                'domain': domain,
                'head_prediction': cached_result['head_prediction'],
                'body_prediction': cached_result['body_prediction'],
                'cached': True
            })

        headers = {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0.2 Mobile/15E148 Safari/604.1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive"
        }

        def fetch(proxies=None):
            try:
                response = requests.get(f"https://{domain}", headers=headers, proxies=proxies, timeout=4)
                if not response.text:
                    response = requests.get(f"http://{domain}", headers=headers, proxies=proxies, timeout=2)
                    if not response.text:
                        response.raise_for_status()
                    return response
                return response
            except requests.exceptions.RequestException as e:
                return "fetch err: "+ str(e)

        # 2. 进行请求
        result = None
        if only_proxy:
            if not socks5_proxy:
                return jsonify({'error': 'socks5_proxy is required when only_proxy=True'}), 400
            proxies = {'http': f'socks5://{socks5_proxy}', 'https': f'socks5://{socks5_proxy}'}
            result = fetch(proxies)
        else:
            result = fetch()
            if isinstance(result, str) and socks5_proxy:
                proxies = {'http': f'socks5://{socks5_proxy}', 'https': f'socks5://{socks5_proxy}'}
                result = fetch(proxies)

        if isinstance(result, str):
            return jsonify({'error': f'Request failed: {result}'}), 500

        response = result
        h_text = [f"{key}: {value}" for key, value in response.headers.items()]
        head_text = "\n".join(h_text)
        body_text = extract_body_text_by_string(response.text)

        # 3. 进行预测
        head_result = predict_text(head_text, head_model, head_tokenizer, HEAD_MAX_LEN)
        body_result = predict_text(body_text, body_model, body_tokenizer, BODY_MAX_LEN)

        # 4. 缓存结果
        set_cache(domain, head_result, body_result)

        return jsonify({
            'domain': domain,
            'head_prediction': head_result,
            'body_prediction': body_result,
            'cached': False
        })

    print(f"Starting API service on port {port}...")
    app.run(host=ip, port=port)

# 命令行调用
if __name__ == "__main__":
    print("device is ", DEVICE)
    parser = argparse.ArgumentParser(description="Train, predict, or serve API using head/body models.")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train_head', 'train_body', 'predict_head', 'predict_body', 'serve_api'],
                        help="Mode: train_head, train_body, predict_head, predict_body, serve_api")
    parser.add_argument('--ok_dir', type=str, default='china-list_out', help="Directory containing ok samples")
    parser.add_argument('--ban_dir', type=str, default='proxy-list_out', help="Directory containing ban samples")
    parser.add_argument('--text', type=str, help="Text to predict (only for predict mode)")
    parser.add_argument('--ip', type=str, default="0.0.0.0", help="ip to serve API (only for serve_api mode)")
    parser.add_argument('--port', type=int, default=5000, help="Port to serve API (only for serve_api mode)")
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'roberta', 'xlm'],
                        help="Model type: bert, roberta-large, xlm-roberta-large")
    parser.add_argument('--load_method', type=str, default='pretrained', choices=['state_dict', 'pretrained'], help="Method to load model for predict: state_dict or pretrained")

    args = parser.parse_args()

    if args.model_type not in model_classes:
        raise ValueError("Invalid model_type. Use 'bert', 'roberta', or 'xlm'.")

    if args.mode == 'train_head':
        train_head_model(args.ok_dir, args.ban_dir, args.model_type)
    elif args.mode == 'train_body':
        train_body_model(args.ok_dir, args.ban_dir, args.model_type)
    elif args.mode == 'serve_api':
        start_api_service(args.ip, args.port, args.model_type, args.load_method)
    elif args.mode in ['predict_head', 'predict_body']:
        model_name = 'head' if args.mode == 'predict_head' else 'body'
        max_len = HEAD_MAX_LEN if model_name == 'head' else BODY_MAX_LEN

        tokenizer_cls, model_cls, model_dir = model_classes[args.model_type]
        
        if args.load_method == 'state_dict':
            tokenizer = tokenizer_cls.from_pretrained(model_dir)
            model = model_cls.from_pretrained(model_dir, num_labels=2).to(DEVICE)
            model.load_state_dict(torch.load(f"best_{model_name}_model.bin"))
        elif args.load_method == 'pretrained':
            model_path = f"{args.model_type}_geosite_by_{model_name}"
            tokenizer = tokenizer_cls.from_pretrained(model_path)
            model = model_cls.from_pretrained(model_path).to(DEVICE)
        else:
            raise ValueError(f"Invalid load_method: {args.load_method}. Use 'state_dict' or 'pretrained'.")
        model.eval()
        
        text = args.text
        if model_name == "body":
            text = extract_body_text_by_string(text)

        result = predict_text(text, model, tokenizer, max_len)
        print(f"Prediction: {result}")
