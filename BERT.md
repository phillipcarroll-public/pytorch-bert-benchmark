# BERT Benchmark

# Create Test Data

```python
import os
import json
import random

# Constants
TRAIN_GOOD_FILE_COUNT = 250
TEST_GOOD_FILE_COUNT = 50
TRAIN_BAD_FILE_COUNT = 250
TEST_BAD_FILE_COUNT = 50
DATA_FOLDER = "data"
WDSDataset_FOLDER = os.path.join(DATA_FOLDER, "benchmark_dataset")

# Generate Good files
good_files = []
for i in range(TRAIN_GOOD_FILE_COUNT + TEST_GOOD_FILE_COUNT):
    data = {
        "id": random.randint(100000, 999999),
        "name": {
            "first": f"John",
            "last": f"Doe {i+1}"
        },
        "age": random.randint(18, 60),
        "score": {
            "math": round(random.uniform(0.6, 1.0), 2),
            "science": round(random.uniform(0.6, 1.0), 2),
            "english": round(random.uniform(0.6, 1.0), 2)
        },
        "address": {
            "street": f"{random.randint(100, 999)} Main St",
            "city": "Anytown",
            "state": "CA",
            "zip": f"{random.randint(10000, 99999)}"
        },
        "contacts": [
            {"type": "email", "value": f"john.doe{i+1}@example.com"},
            {"type": "phone", "value": f"555-{random.randint(1000, 9999)}"}
        ]
    }
    good_files.append(json.dumps(data))

# Generate Bad files
bad_files = []
for i in range(TRAIN_BAD_FILE_COUNT + TEST_BAD_FILE_COUNT):
    data = {
        "id": random.randint(100000, 999999),
        "name": {
            "first": f"John",
            "last": f"Doe {i+1}"
        },
        "age": random.randint(18, 60),
        "score": {
            "math": round(random.uniform(0.4, 0.9), 2),
            "science": round(random.uniform(0.4, 0.9), 2),
            "english": round(random.uniform(0.4, 0.9), 2)
        },
        "address": {
            "street": f"{random.randint(100, 999)} Main St",
            "city": "Anytown",
            "state": "CA",
            "zip": f"{random.randint(10000, 99999)}"
        },
        "contacts": [
            {"type": "email", "value": f"john.doe{i+1}@example.com"},
            {"type": "phone", "value": f"555-{random.randint(1000, 9999)}"}
        ],
        "feedback": ["bad", "not good", "block"][random.randint(0, 2)]
    }
    bad_files.append(json.dumps(data))

# Create train and test folder structure
train_good_folder_path = os.path.join(WDSDataset_FOLDER, "train", "good")
train_bad_folder_path = os.path.join(WDSDataset_FOLDER, "train", "bad")
test_good_folder_path = os.path.join(WDSDataset_FOLDER, "test", "good")
test_bad_folder_path = os.path.join(WDSDataset_FOLDER, "test", "bad")

os.makedirs(train_good_folder_path, exist_ok=True)
os.makedirs(train_bad_folder_path, exist_ok=True)
os.makedirs(test_good_folder_path, exist_ok=True)
os.makedirs(test_bad_folder_path, exist_ok=True)

# Save Good files to train/GOOD and test/GOOD folders
for i in range(TRAIN_GOOD_FILE_COUNT):
    good_file_path = os.path.join(train_good_folder_path, f"good{i+1}.json")
    with open(good_file_path, "w") as f:
        f.write(good_files[i])

for i in range(TEST_GOOD_FILE_COUNT):
    good_file_path = os.path.join(test_good_folder_path, f"good{i+1}.json")
    with open(good_file_path, "w") as f:
        f.write(good_files[TRAIN_GOOD_FILE_COUNT + i])

# Save Bad files to train/BAD and test/BAD folders
for i in range(TRAIN_BAD_FILE_COUNT):
    bad_file_path = os.path.join(train_bad_folder_path, f"bad{i+1}.json")
    with open(bad_file_path, "w") as f:
        f.write(bad_files[i])

for i in range(TEST_BAD_FILE_COUNT):
    bad_file_path = os.path.join(test_bad_folder_path, f"bad{i+1}.json")
    with open(bad_file_path, "w") as f:
        f.write(bad_files[TRAIN_BAD_FILE_COUNT + i])
```

# AMD and NVIDIA

```python
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.amp import GradScaler
import time
import tqdm
import warnings

warnings.filterwarnings("ignore")

# Function to set the number of CPU threads
def set_cpu_thread_cap(num_threads):
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print(f"Set max CPU threads to: {num_threads}")

# Set the device type and device ID
device_type = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0  # Change this to the desired device index, e.g., 0 or 1
# Combine device type and device ID to create the device variable
device = torch.device(f"{device_type}:{device_id}" if device_type else "cpu")
print(f"Device: {device}")
# If running on CPU, set this manually, otherwise comment out
#device = "cpu"

# If running on CPU, set the number of threads
if device == 'cpu':
    set_cpu_thread_cap(30)  # Set this to the desired number of CPU threads


class JSONDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load and preprocess data
        for label in ['good', 'bad']:
            label_dir = os.path.join(data_dir, label)
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(label_dir, file_name)
                    with open(file_path, 'r') as f:
                        script_data = json.load(f)
                        # Convert JSON data to a string (simple approach, can be improved)
                        text = str(script_data)

                        self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Convert to tensors
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(1 if label == 'good' else 0, dtype=torch.long)
        }

        return item

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move the model to the device
model.to(device)

# Dataset and DataLoader
train_dir = os.path.join("data", "benchmark_dataset", "train")
test_dir = os.path.join("data", "benchmark_dataset", "test")

train_dataset = JSONDataset(train_dir, tokenizer)
test_dataset = JSONDataset(test_dir, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Initialize GradScaler for mixed precision training
scaler = torch.GradScaler(device_type)

# Training function
def train_epoch(model, train_loader, optimizer, scaler):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Testing function
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['bad', 'good'], labels=[0, 1])

    return accuracy, report

# Training loop
epochs = 1

benchmarkLoops = 10

# Start timer
start_time = time.time()

# Run the training loop with tqdm for progress bar
for i in tqdm.tqdm(range(benchmarkLoops)):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scaler)
        accuracy, report = test_model(model, test_loader)
        # print time and loop number
        print(f"Loop {i+1} - Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Accuracy: {accuracy:.4f}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
```

# Intel XPU

```python
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.amp import GradScaler
# Import IPEX
import intel_extension_for_pytorch as ipex
import time
import tqdm
import warnings

# Clear cache
torch.xpu.empty_cache()

warnings.filterwarnings("ignore")

# Function to set the number of CPU threads
def set_cpu_thread_cap(num_threads):
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print(f"Set max CPU threads to: {num_threads}")

# Set the device type and device ID
device_type = "xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0  # Change this to the desired device index, e.g., 0 or 1
# Combine device type and device ID to create the device variable
device = torch.device(f"{device_type}:{device_id}" if device_type else "cpu")
print(f"Device: {device}")
# If running on CPU, set this manually, otherwise comment out
#device = "cpu"

# If running on CPU, set the number of threads
if device == 'cpu':
    set_cpu_thread_cap(8)  # Set this to the desired number of CPU threads


class JSONDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        # Load and preprocess data
        for label in ['good', 'bad']:
            label_dir = os.path.join(data_dir, label)
            for file_name in os.listdir(label_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(label_dir, file_name)
                    with open(file_path, 'r') as f:
                        script_data = json.load(f)
                        # Convert JSON data to a string (simple approach, can be improved)
                        text = str(script_data)

                        self.data.append((text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Convert to tensors
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(1 if label == 'good' else 0, dtype=torch.long)
        }

        return item

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Move the model to the device
model.to(device)
# Compile model
model = torch.compile(model) 

# Dataset and DataLoader
train_dir = os.path.join("data", "benchmark_dataset", "train")
test_dir = os.path.join("data", "benchmark_dataset", "test")

train_dataset = JSONDataset(train_dir, tokenizer)
test_dataset = JSONDataset(test_dir, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)



# Training function
def train_epoch(model, train_loader, optimizer):
    model.train()

    # Optimize the model with ipex.optimizer
    # This must be done when the model is set to train
    model, optimizer = ipex.optimize(model, optimizer=optimizer)
    
    total_loss = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# Testing function
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.amp.autocast(device_type):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=['bad', 'good'], labels=[0, 1])

    return accuracy, report

# Training loop
epochs = 1

benchmarkLoops = 10

# Start timer
start_time = time.time()

# Run the training loop with tqdm for progress bar
for i in tqdm.tqdm(range(benchmarkLoops)):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        accuracy, report = test_model(model, test_loader)
        # print time and loop number
        print(f"Loop {i+1} - Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Accuracy: {accuracy:.4f}")

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Clear cache and vram
del model, tokenizer
torch.xpu.empty_cache()
```

## Dual GPU DistributedDataParallel Training ROCm (AMD)

```
import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# --- Your Dataset Class ---
class JSONDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data = []
        # Synthetic data generation if directories don't exist for testing
        if not os.path.exists(data_dir):
            for _ in range(100):
                self.data.append(("dummy text content", "good"))
        else:
            for label in ['good', 'bad']:
                label_dir = os.path.join(data_dir, label)
                if os.path.exists(label_dir):
                    for file_name in os.listdir(label_dir):
                        if file_name.endswith('.json'):
                            self.data.append(("script text content here", label))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(1 if label == 'good' else 0, dtype=torch.long)
        }

def setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    rank = setup()
    device = torch.device(f"cuda:{rank}")
    
    # Model Setup
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    model = DDP(model, device_ids=[rank])
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler("cuda")

    # Data Setup
    train_dataset = JSONDataset("data/benchmark_dataset/train", tokenizer)
    sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=4, sampler=sampler)

    # Benchmark Parameters
    benchmark_loops = 10
    
    # Synchronization point before starting timer
    dist.barrier()
    start_time = time.perf_counter()

    for i in range(benchmark_loops):
        sampler.set_epoch(i)
        model.train()
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if rank == 0:
            print(f"Loop {i+1}/{benchmark_loops} complete.")

    # Synchronization point before ending timer
    dist.barrier()
    end_time = time.perf_counter()

    if rank == 0:
        total_time = end_time - start_time
        print(f"\n{'='*30}")
        print(f"BENCHMARK COMPLETE")
        print(f"Total time for {benchmark_loops} loops: {total_time:.2f} seconds")
        print(f"Average time per loop: {total_time/benchmark_loops:.2f} seconds")
        print(f"{'='*30}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

Save the above file as `test.py`

Run with: `torchrun --nproc_per_node=2 test.py`
