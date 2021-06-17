# Academic Network Embedding & Downstream Applications

这里是EE226大数据挖掘，学期项目学术网络嵌入及其应用的README

## Reference

- 对于同构图嵌入模型，借鉴了 https://github.com/shenweichen/GraphEmbedding，修改后得到对本项目的应用，主要实现了DeepWalk, node2vec, SDNE 等模型
- 对于异构图嵌入模型，借鉴了 https://github.com/stellargraph/stellargraph/blob/develop/demos/embeddings/metapath2vec-embeddings.ipynb，修改后实现了metapath2vec模型对本项目的应用。

*注：本项目中的数据处理，其余方法技巧，以及downstream的节点分类和链路预测模型和算法都是独立实现的*

## Setup

由于时间问题，暂未将模型环境统一：

对于同构图模型，conda构建虚拟环境1：

- python=3.6
- tensorflow>=1.4.0,<=1.12.0
- gensim==3.6.0
- networkx==2.1
- joblib==0.13.0
- fastdtw==0.3.2
- tqdm
- numpy
- scikit-learn
- pandas
- matplotlib

对于异构图模型，conda构建虚拟环境2：

```
pip install stellargraph[demos]
```

*注：对节点分类深度模型的测试 还需要tensorflow*

## How to test

- deepwalk.py, node2vec.py, sdne.py 实现了同构图DeepWalk, node2vec以及SDNE模型对本项目的应用，采用**环境1**，可直接运行
- metapath2vec.py 实现了异构图metapath2vec模型对本项目的应用，采用**环境2**，可直接运行
- DL_multilabel.py 实现了基于embedding进行deep learning得到多标签分类，需要运行metapath2vec.py的到embedding后再运行，且需要注意输出embedding array命名的一致性

## Other codes introduction

- *data_preprocessing*文件夹中包含了我们对于数据的预处理
- classification1/2.py 实现了基于scikit-learn的传统多标签分类器
- graph.py 用于构建异构图网络 联合metapath2vec.py使用
- link_pred.py 实现了链路预测
- node_class.py 基于author embedding多标签分类 其中调用了classification2.py
- paper_class.py 基于paper embedding分类 再映射到author多标签上 其中调用了classification2.py
- visualization.py 对于我们得到的embedding进行可视化

## Other files introduction

- ge文件夹中包含了各种embedding模型本身的生成
- models 输出文件夹，用来存放多标签分类的深度模型参数
- embeddings_array 输出文件夹， 用来存放得到的embedding 格式为np.array
- data 输入文件夹，存放着大部分其他代码所需的数据输入

## Contributors

- Yifei [ liyifei919518@sjtu.edu.cn](mailto:liyifei919518@sjtu.edu.cn)
- Haoning Wu [whn15698781666@sjtu.edu.cn](mailto:whn15698781666@sjtu.edu.cn)
- Longrun Zhi [zlongrun@sjtu.edu.cn](mailto:zlongrun@sjtu.edu.cn)