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

Run the code with:

```
python assignment1/__init__.py
```

You can change the experiment arguements in `assignment1/config.yaml`

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

You can change the experiment arguements in `assignment1/config.yaml`

You can check the progress by:

```
bjobs
```
