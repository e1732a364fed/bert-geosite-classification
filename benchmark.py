import time
import torch
from classify import predict_text

# Sample text for testing
sample_text = "<body>googlegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegooglegoogle</body>"

BERT = "./bert_geosite_by_body/"


def benchmark_classification(device_type):
    """Run classification on the specified device and return average execution time over 100 runs."""
    current_device = torch.device(device_type)
    print("running benchmark for ", device_type)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BERT)
    model = AutoModelForSequenceClassification.from_pretrained(BERT, num_labels=2).to(
        current_device
    )

    total_time = 0.0

    # Run the test 100 times and calculate average execution time
    for _ in range(100):
        start_time = time.time()

        result = predict_text(sample_text, model, tokenizer, 512, current_device)

        end_time = time.time()
        execution_time = end_time - start_time
        total_time += execution_time

    average_time = total_time / 100

    return average_time, result


def main():
    cpu_time, cpu_result = benchmark_classification("cpu")

    mps_time, mps_result = benchmark_classification("mps")

    print("\n=== Classification Benchmark Results ===")
    print(f"CPU Execution Time: {cpu_time:.4f} seconds")
    print(f"CPU Result: {cpu_result}")

    if mps_time is not None:
        print(f"MPS Execution Time: {mps_time:.4f} seconds")
        print(f"MPS Result: {mps_result}")

        # Calculate speedup
        if cpu_time > 0:
            speedup = (cpu_time - mps_time) / cpu_time * 100
            print(f"\nMPS is {speedup:.2f}% faster than CPU")
    else:
        print("\nMPS is not available on this system.")


if __name__ == "__main__":
    main()
