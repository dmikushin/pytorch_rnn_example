# Testing PyTorch with MIOpen backend

Testing PyTorch with MIOpen backend, based on [Introduction to RNNs](https://medium.com/swlh/introduction-to-recurrent-neural-networks-rnns-347903dd8d81) by Tim Sullivan.

## Prerequisites

Setup MIOpen-tailored PyTorch in a virtual environment:

```
python3 -m venv ./venv
source venv/bin/activate.fish
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2
```

## Running

```
python3 pytorch_rnn_example.py 
```
