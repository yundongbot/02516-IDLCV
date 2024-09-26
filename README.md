# 02516IDLCV

## Prerequisites

- Python 3.10+

### Run on the local environment

Install the requirements with:

```
pip install -r requirements.txt
```

Then you need to download and unzip the dataset to `assignment1` folder:

```
wget https://huggingface.co/datasets/yununuy/hotdot_nohotdog/resolve/main/hotdog_nothotdog.zip
unzip hotdog_nothotdog.zip -d ./assignment1
```

### Run on the HPC:

Clone code to your hpc instance:

```
git clone git@github.com:yundongbot/02516-IDLCV.git
```

Then you need to download and unzip the dataset to `assignment1` folder:

```
cd 02516-IDLCV
wget https://huggingface.co/datasets/yununuy/hotdot_nohotdog/resolve/main/hotdog_nothotdog.zip
unzip hotdog_nothotdog.zip -d ./assignment1
```

Then submit the job:

```
bsub < ./scripts/assignment1.sub
```

You can change the experiment arguements by commandline:

```
bsub < ./scripts/assignment1.sub Resnet18 SGD 0.0002 100 128
```

```
parser.add_argument('--model', type=str, default='VGG16', help='Model name (e.g., VGG16, Resnet18)')
parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name (default: Adam)')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs (default: 100)')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
```

You can check the progress by:

```
bjobs
```

# Assignment 1

Run the code with:

```
python assignment1/__init__.py
```
