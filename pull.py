import os
import requests
import argparse
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_processed_domains(output_dir):
    """加载已处理的域名列表"""
    processed_file = os.path.join(output_dir, 'processed.txt')
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            return set(f.read().splitlines())
    return set()

def save_processed_domain(output_dir, domain):
    """保存已处理的域名"""
    processed_file = os.path.join(output_dir, 'processed.txt')
    with open(processed_file, 'a') as f:
        f.write(domain + '\n')

def save_error(output_dir, domain, error_message):
    """保存错误信息"""
    error_file = os.path.join(output_dir, 'errors.txt')
    with open(error_file, 'a', encoding='utf-8') as f:
        f.write(f"{domain}    {error_message}\n")

def save_response(output_dir, domain, headers, body):
    """保存HTTP响应的头部和主体"""
    safe_domain = quote(domain, safe="")
    domain_dir = os.path.join(output_dir, safe_domain)
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, 'head.txt'), 'w', encoding='utf-8') as f:
        for key, value in headers.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(domain_dir, 'body.txt'), 'w', encoding='utf-8') as f:
        f.write(body)

def process_domain(output_dir, domain):
    """处理单个域名"""
    isfull = domain.startswith("full:")
    domain = domain.removeprefix("full:")
    
    urls = [f"https://{domain}", f"https://www.{domain}", f"http://{domain}", f"http://www.{domain}"] if not isfull else [f"https://{domain}", f"http://{domain}"]
    
    headers = {
        "User Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0.2 Mobile/15E148 Safari/604.1",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive"
    }
    for url in urls:
        try:
            response = requests.get(url, headers=headers, timeout=4)
            if response.text:
                save_response(output_dir, domain, response.headers, response.text)
                return  # 只要有一个成功就返回
        except Exception as e:
            error_msg = str(e)
    
    # 所有请求失败才记录错误
    save_error(output_dir, domain, error_msg if 'error_msg' in locals() else f"HTTP {response.status_code}")

def main(list_file, output_dir):
    """主函数 - 并行处理多个域名"""
    processed_domains = load_processed_domains(output_dir)
    
    with open(list_file, 'r') as f:
        domains = [line.strip() for line in f if line.strip() and line.strip() not in processed_domains and not line.strip().startswith("regexp:")]

    if not domains:
        print("所有域名已处理,无需请求。")
        return

    # 限制并发数量,避免过载
    max_workers = min(20, len(domains))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_domain = {executor.submit(process_domain, output_dir, domain): domain for domain in domains}

        for future in as_completed(future_to_domain):
            domain = future_to_domain[future]
            try:
                future.result()  # 触发异常(如果有的话)
                save_processed_domain(output_dir, domain)
            except Exception as e:
                save_error(output_dir, domain, f"线程错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch HTTP responses for a list of domains.")
    parser.add_argument('-l', '--list-file', default='china-list.txt', help='Path to the list file containing domains.')
    args = parser.parse_args()

    output_dir = os.path.join(os.getcwd(), f"{os.path.basename(os.path.splitext(args.list_file)[0])}_out")
    os.makedirs(output_dir, exist_ok=True)

    main(args.list_file, output_dir)
