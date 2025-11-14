# Running TorchBrain models for the FALCON benchmark.
The following commands expect to be run from `./examples/falcon` as working directory.

### Installation

```bash
conda create -n torchbrain python=3.11
pip install pytorch_brain lightning==2.3.3 wandb~=0.15 falcon-challenge
```

### Training an RNN baseline.
The FALCON datasets have been prepared for `torch_brain` using `brainsets`.
```bash
brainsets prepare falcon_m2
```

Train an RNN for this dataset by running:

```bash
python train.py --config-name train_rnn_m2.yaml
```

### Prepare a dockerfile for submission

Symlink or link the data directory to where the local docker code expects it:

```bash
ln -s /path/to/torchbrain/raw/falcon_h1_2024 ./data/h1
ln -s /path/to/torchbrain/raw/falcon_m1_2024 ./data/m1
ln -s /path/to/torchbrain/raw/falcon_m2_2024 ./data/m2
```

# JY TODO
```
# Build
docker build -t torchbrain -f ./Dockerfile .
bash test_docker_local.sh --docker-name torchbrain:latest
```

###
EvalAI CLI is bugged on python 3.11, create a separate environment for it.
```bash
conda create -n evalai python=3.10
conda activate evalai
pip install evalai
```

# Sign in
```bash
# Register an evalai account and sign up for the FALCON challenge.
# https://eval.ai/web/challenges/challenge-page/2319/

# Register your CLI
evalai set_token <your_token> # Available on https://eval.ai/web/challenges/challenge-page/2319/submission or in account settings

# Push
evalai push torchbrain:latest --phase few-shot-test-2319 --private
```