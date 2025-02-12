import os
import requests
import argparse

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
    with open(error_file, 'a') as f:
        f.write(f"{domain}    {error_message}\n")

def save_response(output_dir, domain, headers, body):
    """保存HTTP响应的头部和主体"""
    domain_dir = os.path.join(output_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    with open(os.path.join(domain_dir, 'head.txt'), 'w') as f:
        for key, value in headers.items():
            f.write(f"{key}: {value}\n")
    with open(os.path.join(domain_dir, 'body.txt'), 'w', encoding='utf-8') as f:
        f.write(body)

def process_domain(output_dir, domain):
    """处理单个域名"""
    try:
        response = requests.get(f"http://{domain}", timeout=10)
        save_response(output_dir, domain, response.headers, response.text)
    except Exception as e:
        save_error(output_dir, domain, str(e))

def main(list_file, output_dir):
    """主函数"""
    processed_domains = load_processed_domains(output_dir)
    with open(list_file, 'r') as f:
        for line in f:
            domain = line.strip()
            if domain in processed_domains:
                continue
            process_domain(output_dir, domain)
            save_processed_domain(output_dir, domain)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch HTTP responses for a list of domains.")
    parser.add_argument('-l', '--list-file', default='china-list.txt', help='Path to the list file containing domains.')
    args = parser.parse_args()

    list_file = args.list_file
    output_dir = f"{os.path.splitext(list_file)[0]}_out"
    os.makedirs(output_dir, exist_ok=True)

    main(list_file, output_dir)
