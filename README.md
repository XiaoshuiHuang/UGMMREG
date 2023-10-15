# Implementation of [Unsupervised Point Cloud Registration by Learning Unified Gaussian Mixture Models](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9790333)


## Requirements

To run this codebase, [PyTorch](https://pytorch.org/get-started/locally/) is required. 

Then install the required packages by running the following command:

```bash
pip install -r requirements.txt
```

## Data Preprocessing
To prepare the dataset, run the following command:
```bash
python ./data/data_prepare.py
```

## Usage
### Training
To train the model, run the following command:
```bash
python main.py train --dataset_path ./dataset/ModelNet40TrainVal
```

### Testing
For testing, use the following command and specify the checkpoint path:
```bash
python main.py test --dataset_path ./dataset/ModelNet40 --test_ckpt_path {TEST_CKPT_PATH}
```

A pretrained model is also available at `./checkpoints/model.pth.`

## Acknowledgements
Our code is built upon various repositories including [FMR](https://github.com/XiaoshuiHuang/fmr), [DeepGMR](https://github.com/wentaoyuan/deepgmr), and [JRMPC](https://team.inria.fr/perception/research/jrmpc/).

## Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{huang2022unsupervised,
  title={Unsupervised point cloud registration by learning unified gaussian mixture models},
  author={Huang, Xiaoshui and Li, Sheng and Zuo, Yifan and Fang, Yuming and Zhang, Jian and Zhao, Xiaowei},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={3},
  pages={7028--7035},
  year={2022},
  publisher={IEEE}
}
```