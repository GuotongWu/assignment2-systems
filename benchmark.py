import argparse
import timeit
import torch 
import numpy as np
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

args = argparse.ArgumentParser(description="Benchmark the BasicsTransformerLM model.")

args.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
args.add_argument("--context_length", type=int, default=256, help="Length of the context.")
args.add_argument("--d_model", type=int, default=768, help="Dimension of the model.")
args.add_argument("--num_layers", type=int, default=12, help="Number of transformer layers.")
args.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
args.add_argument("--d_ff", type=int, default=3072, help="Dimension of the feedforward network.")
args.add_argument("--rope_theta", type=float, default=10000.0, help="Theta value for RoPE positional encoding.")
args.add_argument("--batch_size", type=int, default=4, help="Batch size")
args.add_argument("--num_steps", type=int, default=10, help="The total steps for execution")
args.add_argument("--type", type=str, default="f", help="forward or forward+backward")
args.add_argument("--num_warmups", type=int, default=5, help="The total steps for warm-up")

args = args.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
    
device = get_device()

model = BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta
).to(device)
optimizer = AdamW(model.parameters())

input_data = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length), device=device)
target = torch.randint(low=0, high=args.vocab_size, size=(args.batch_size, args.context_length), device=device)

def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if torch.mps.is_available():
        torch.mps.synchronize()

def run_model(model, input_data, target, type):
    out = model(input_data)
    if type == "fb":
        loss = cross_entropy(out, target)
        loss.backward()
        model.zero_grad()
        optimizer.step()
    sync_device()

for _ in range(args.num_warmups):
    run_model(model, input_data, target, args.type)

sync_device()

step_time = []
# Start recording memory history.
torch.cuda.memory._record_memory_history(max_entries=1000000)
for _ in range(args.num_steps):
    start = timeit.default_timer()
    run_model(model, input_data, target, args.type)
    end = timeit.default_timer()
    step_time.append(end - start)

# Save a pickle file to be loaded by PyTorch's online tool.
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Stop recording history.
torch.cuda.memory._record_memory_history(enabled=None)

print(np.mean(step_time), np.std(step_time))