import torch

def main():
    print("Hello from cs2!")
    print("--- GPU CONNECTION TEST ---")

    if torch.cuda.is_available():
        print("SUCCESS! A GPU was detected.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    else:
        print("FAILED. PyTorch cannot find a GPU.")

if __name__ == "__main__":
    main()

