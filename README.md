# Interactive Language Learning by Question Answering
--------------------------------------------------------------------------------
Code for EMNLP 2019 paper "Interactive Language Learning by Question Answering".

## To install dependencies
```
sudo apt update
conda create -p ~/venvs/qait python=3.6
source activate ~/venvs/qait
pip install --upgrade pip
pip install numpy==1.16.4
pip install https://github.com/Microsoft/TextWorld/archive/rebased-interactive-qa.zip
pip install -U spacy
python -m spacy download en
pip install tqdm h5py visdom pyyaml
conda install pytorch torchvision cudatoolkit=9.2 -c pytorch
```

## Test Set
Download the test set from [https://aka.ms/qait-testset](https://aka.ms/qait-testset). Unzip it.


## Pretrained Word Embeddings
Before first time running it, download fasttext crawl-300d-2M.vec.zip from [HERE](https://fasttext.cc/docs/en/english-vectors.html), unzip, and run [embedding2h5.py](./embedding2h5.py) for fast embedding loading in the future.

## To Train
```
python train.py ./
```

## Citation

Please use the following bibtex entry:
```
@article{yuan2019qait,
  title={Interactive Language Learning by Question Answering},
  author={Yuan, Xingdi and C\^ot\'{e}, Marc-Alexandre and Fu, Jie and Lin, Zhouhan and Pal, Christopher and Bengio, Yoshua and Trischler, Adam},
  booktitle={EMNLP},
  year={2019}
}
```

## License

[MIT](./LICENSE)
