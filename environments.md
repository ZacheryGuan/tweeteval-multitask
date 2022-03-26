## 装载环境和安装依赖
**在discovery node (ssh)中**

module load python/3.7 cuda/11.0

pip install fasttext transformers nltk

python -m ipykernel install --user --name proj --display-name "py37_torch17_cuda110"

torch 1.7 + cu110 success:
```shell
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## 准备数据
在与notebook相同的目录下:
```shell
git clone https://github.com/cardiffnlp/tweeteval.git
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gzip -d cc.en.300.bin.gz && rm cc.en.300.bin.gz

# other notes
```

k40 doesn't support torch16(card too old)

a100 doesn't support sm75
torch1.6 + cu102
```
NVIDIA A100-PCIE-40GB with CUDA capability sm_80 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75.
If you want to use the NVIDIA A100-PCIE-40GB GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

```
torch 1.8 + cu102
```python
NVIDIA A100-PCIE-40GB with CUDA capability sm_80 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
```


