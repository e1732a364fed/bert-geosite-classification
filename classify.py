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

# 配置参数
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

print("device is ", DEVICE)

# 固定随机种子
torch.manual_seed(SEED)
np.random.seed(SEED)

# 数据收集函数
def collect_samples(ok_dir, ban_dir):
    print("collect_samples")
    samples = []
    # 收集正常样本
    for domain in os.listdir(ok_dir):
        domain_path = os.path.join(ok_dir, domain)
        head_path = os.path.join(domain_path, "head.txt")
        body_path = os.path.join(domain_path, "body.txt")
        if os.path.exists(head_path) and os.path.exists(body_path):
            try:
                with open(head_path, 'r', encoding='utf-8', errors='ignore') as f:
                    head_text = f.read().strip()
                with open(body_path, 'r', encoding='utf-8', errors='ignore') as f:
                    body_text = f.read().strip()
                samples.append({
                    "head": head_text,
                    "body": body_text,
                    "label": 0
                })
            except Exception as e:
                print(f"Error reading files for domain {domain}: {e}")

    # 收集封禁样本
    for domain in os.listdir(ban_dir):
        domain_path = os.path.join(ban_dir, domain)
        head_path = os.path.join(domain_path, "head.txt")
        body_path = os.path.join(domain_path, "body.txt")
        if os.path.exists(head_path) and os.path.exists(body_path):
            try:
                with open(head_path, 'r', encoding='utf-8', errors='ignore') as f:
                    head_text = f.read().strip()
                with open(body_path, 'r', encoding='utf-8', errors='ignore') as f:
                    body_text = f.read().strip()
                samples.append({
                    "head": head_text,
                    "body": body_text,
                    "label": 1
                })
            except Exception as e:
                print(f"Error reading files for domain {domain}: {e}")
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
        logits = outputs.logits.cpu().numpy()
        pred = np.argmax(logits, axis=1)[0]

    return "ok" if pred == 0 else "ban"


# 训练Head模型
def train_head_model(ok_dir, ban_dir, model_type):
    samples = collect_samples(ok_dir, ban_dir)
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, stratify=[s['label'] for s in samples], random_state=SEED)
    
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
        model = BertForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2).to(DEVICE)
        
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_DIR)
        model = RobertaForSequenceClassification.from_pretrained(ROBERTA_DIR, num_labels=2).to(DEVICE)
        
    elif model_type == 'xlm':
        tokenizer = AutoTokenizer.from_pretrained(XLM_ROBERTA)
        model = AutoModelForSequenceClassification.from_pretrained(XLM_ROBERTA, num_labels=2).to(DEVICE)
        
    else:
        raise ValueError("Invalid model_type. Use 'bert', ROBERTA_DIR, or XLM_ROBERTA.")
    
    train_dataset = TextDataset(train_samples, tokenizer, HEAD_MAX_LEN, 'head')
    val_dataset = TextDataset(val_samples, tokenizer, HEAD_MAX_LEN, 'head')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    train_model(model, train_loader, val_loader, "best_head_model.bin")

# 训练Body模型
def train_body_model(ok_dir, ban_dir, model_type):
    samples = collect_samples(ok_dir, ban_dir)
    train_samples, val_samples = train_test_split(
        samples, test_size=0.2, stratify=[s['label'] for s in samples], random_state=SEED)
    
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
        model = BertForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2).to(DEVICE)
        
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_DIR)
        model = RobertaForSequenceClassification.from_pretrained(ROBERTA_DIR, num_labels=2).to(DEVICE)
        
    elif model_type == 'xlm':
        tokenizer = AutoTokenizer.from_pretrained(XLM_ROBERTA)
        model = AutoModelForSequenceClassification.from_pretrained(XLM_ROBERTA, num_labels=2).to(DEVICE)
        
    else:
        raise ValueError("Invalid model_type. Use 'bert', ROBERTA_DIR, or XLM_ROBERTA.")
    
    train_dataset = TextDataset(train_samples, tokenizer, BODY_MAX_LEN, 'body')
    val_dataset = TextDataset(val_samples, tokenizer, BODY_MAX_LEN, 'body')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    train_model(model, train_loader, val_loader, "best_body_model.bin")


# API服务函数
def start_api_service(port, model_type):
    app = Flask(__name__)

    # 选择模型和 tokenizer
    model_paths = {
        "head": "best_head_model.bin",
        "body": "best_body_model.bin"
    }

    if model_type == 'bert':
        tokenizer_cls = BertTokenizer
        model_cls = BertForSequenceClassification
        
    elif model_type == 'roberta':
        tokenizer_cls = RobertaTokenizer
        model_cls = RobertaForSequenceClassification
        
    elif model_type == 'xlm':
        tokenizer_cls = AutoTokenizer
        model_cls = AutoModelForSequenceClassification
        
    else:
        raise ValueError("Invalid model_type. Use 'bert', ROBERTA_DIR, or XLM_ROBERTA.")

    head_tokenizer = tokenizer_cls.from_pretrained(model_type)
    body_tokenizer = tokenizer_cls.from_pretrained(model_type)

    head_model = model_cls.from_pretrained(model_type, num_labels=2).to(DEVICE)
    body_model = model_cls.from_pretrained(model_type, num_labels=2).to(DEVICE)

    head_model.load_state_dict(torch.load(model_paths["head"]))
    body_model.load_state_dict(torch.load(model_paths["body"]))

    head_model.eval()
    body_model.eval()

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        text = data.get('text', '')
        model_type = data.get('model_type', 'head')  # 默认为head模型

        if model_type not in ['head', 'body']:
            return jsonify({'error': 'Invalid model_type. Use "head" or "body".'}), 400

        model = head_model if model_type == 'head' else body_model
        tokenizer = head_tokenizer if model_type == 'head' else body_tokenizer
        max_len = HEAD_MAX_LEN if model_type == 'head' else BODY_MAX_LEN

        result = predict_text(text, model, tokenizer, max_len)
        return jsonify({'result': result})

    print(f"Starting API service on port {port}...")
    app.run(host='0.0.0.0', port=port)

# 命令行调用
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, predict, or serve API using head/body models.")
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train_head', 'train_body', 'predict_head', 'predict_body', 'serve_api'],
                        help="Mode: train_head, train_body, predict_head, predict_body, serve_api")
    parser.add_argument('--ok_dir', type=str, default='china-list_out', help="Directory containing ok samples")
    parser.add_argument('--ban_dir', type=str, default='proxy-list_out', help="Directory containing ban samples")
    parser.add_argument('--text', type=str, help="Text to predict (only for predict mode)")
    parser.add_argument('--port', type=int, default=5000, help="Port to serve API (only for serve_api mode)")
    parser.add_argument('--model_type', type=str, default='bert', choices=['bert', 'roberta', 'xlm'],
                        help="Model type: bert, roberta-large, xlm-roberta-large")
    args = parser.parse_args()

    if args.mode == 'train_head':
        train_head_model(args.ok_dir, args.ban_dir, args.model_type)
    elif args.mode == 'train_body':
        train_body_model(args.ok_dir, args.ban_dir, args.model_type)
    elif args.mode == 'serve_api':
        start_api_service(args.port, args.model_type)
    elif args.mode in ['predict_head', 'predict_body']:
        model_path = "best_head_model.bin" if args.mode == 'predict_head' else "best_body_model.bin"
        model_type = 'head' if args.mode == 'predict_head' else 'body'
        max_len = HEAD_MAX_LEN if model_type == 'head' else BODY_MAX_LEN

        if args.model_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
            model = BertForSequenceClassification.from_pretrained(BERT_DIR, num_labels=2).to(DEVICE)
        elif args.model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_DIR)
            model = RobertaForSequenceClassification.from_pretrained(ROBERTA_DIR, num_labels=2).to(DEVICE)
        elif args.model_type == 'xlm':
            tokenizer = AutoTokenizer.from_pretrained(XLM_ROBERTA)
            model = AutoModelForSequenceClassification.from_pretrained(XLM_ROBERTA, num_labels=2).to(DEVICE)
        else:
            raise ValueError("Invalid model_type. Use 'bert', ROBERTA_DIR, or XLM_ROBERTA.")

        model.load_state_dict(torch.load(model_path))
        model.eval()

        result = predict_text(args.text, model, tokenizer, max_len)
        print(f"Prediction: {result}")
