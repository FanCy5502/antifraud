# AntiFraud (反欺诈)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-amazon-fraud)](https://paperswithcode.com/sota/fraud-detection-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-amazon-fraud)](https://paperswithcode.com/sota/node-classification-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-yelp-fraud)](https://paperswithcode.com/sota/fraud-detection-on-yelp-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-yelpchi)](https://paperswithcode.com/sota/node-classification-on-yelpchi?p=semi-supervised-credit-card-fraud-detection)

AntiFraud是一个金融欺诈检测框架。

已复现的论文模型：
- `MCNN`: 使用卷积神经网络进行信用卡欺诈检测，发表于 ICONIP 2016。
- `STAN`: 基于时空注意力的神经网络进行信用卡欺诈检测，发表于 AAAI 2020。
- `STAGN`: 基于时空注意力的图神经网络用于欺诈检测，发表于 TKDE 2020。
- `GTAN`: 基于属性驱动的图表征进行半监督信用卡欺诈检测，发表于 AAAI 2023。
- `RGTAN`: 使用风险感知的图表征增强属性驱动的欺诈检测。
- `HOGRL`: 高阶图表征学习在信用卡欺诈检测中的有效性。

## 使用说明

### 数据处理
1. 通过执行命令 `unzip /data/Amazon.zip` 和 `unzip /data/YelpChi.zip` 解压数据集；
2. 通过执行命令 `python feature_engineering/data_process.py` 预处理本仓库所需的所有数据集。
3. 通过执行命令 `python feature_engineering/get_matrix.py` 来生成高阶交易图的邻接矩阵。请注意，这需要大约 280GB 的存储空间。如果您打算运行 `HOGRL`模型，应首先执行 `get_matrix.py` 脚本。

### 训练与评估
运行 `MCNN`、`STAN` 和 `STAGN` 模型的命令为：
```
python main.py --method mcnn
python main.py --method stan
python main.py --method stagn
```
有关超参数的详细说明，请参考 `config/gtan_cfg.yaml` 和 `config/rgtan_cfg.yaml`。

运行`HOGRL` 模型的命令为：
```
python main.py --method hogrl
```
有关超参数的详细说明，请参考 `config/hogrl_cfg.yaml`。

### 数据描述

本仓库基于三个数据集，YelpChi、Amazon 和 S-FFSD，进行模型实验。

YelpChi 和 Amazon 数据集来自 [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903)，原始数据可在 [该仓库](https://github.com/YingtongDou/CARE-GNN/tree/master/data) 中找到。

S-FFSD 是一个模拟的小型金融欺诈半监督数据集。S-FFSD 的数据描述如下：

| 名称  | 类型  | 范围  | 说明  |
|-------|-------|-------|-------|
| Time  | np.int32 | 从 $\mathbf{0}$ 到 $\mathbf{N}$ | $\mathbf{N}$ 表示交易的数量。|
| Source| string | 从 $\mathbf{S_0}$ 到 $\mathbf{S_{ns}}$ | $ns$ 表示交易发送者的数量。|
| Target| string | 从 $\mathbf{T_0}$ 到 $\mathbf{T_{nt}}$ | $nt$ 表示交易接收者的数量。|
| Amount| np.float32 | 从 **0.00** 到 **np.inf** | 每笔交易的金额。|
| Location| string | 从 $\mathbf{L_0}$ 到 $\mathbf{L_{nl}}$ | $nl$ 表示交易地点的数量。|
| Type  | string | 从 $\mathbf{TP_0}$ 到 $\mathbf{TP_{np}}$ | $np$ 表示不同交易类型的数量。|
| Labels| np.int32 | 从 **0** 到 **2** | **2** 表示 **未标注** |

> 我们正在寻找有趣的公开数据集！如果您有任何建议，欢迎联系我们！

## 测试结果

以下是五个模型在三个数据集上的表现：
| |YelpChi| | |Amazon| | |S-FFSD| | |
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| |AUC|F1|AP|AUC|F1|AP|AUC|F1|AP|
|MCNN||- | -| -| -| -|0.7129|0.6861|0.3309|
|STAN|- |- | -| -| -| -|0.7446|0.6791|0.3395|
|STAGN|- |- | -| -| -| -|0.7659|0.6852|0.3599|
|GTAN|0.9241|0.7988|0.7513|0.9630|0.9213|0.8838|0.8286|0.7336|0.6585|
|RGTAN|0.9498|0.8492|0.8241|0.9750|0.9200|0.8926|0.8461|0.7513|0.6939|
|HOGRL|0.9808|0.8595|-|0.9800|0.9198|-|-|-|-|

> `MCNN`、`STAN` 和 `STAGN` 目前不适用于 YelpChi 和 Amazon 数据集。
>
> `HOGRL` 目前不适用于 S-FFSD 数据集。

## 仓库结构

本仓库结构如下：
- `models/`: 每种方法的预训练模型。可以自行训练模型或直接使用我们提供的预训练模型；
- `data/`: 数据集文件；
- `config/`: 不同模型的配置文件；
- `feature_engineering/`: 数据处理；
- `methods/`: 模型实现；
- `main.py`: 组织项目整体的配置加载、数据处理和模型运行；
- `requirements.txt`: 包依赖；

## 环境要求
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.8.1
```
### 贡献者：
<a href="https://github.com/AI4Risk/antifraud/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=AI4Risk/antifraud" />
</a>

### 引用

如果 Antifraud 对您的研究有所帮助，请考虑引用以下文献：
    
    @inproceedings{zou2024effective,
      title={Effective High-order Graph Representation Learning for Credit Card Fraud Detection.},
      author={Zou, Yao and Cheng, Dawei},
      booktitle={International Joint Conference on Artificial Intelligence},
      year={2024}
    }
    @inproceedings{Xiang2023SemiSupervisedCC,
        title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
        author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }

