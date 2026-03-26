# BoW / PLM Graph Experiments

这个项目用于复现实验表格中的节点分类结果，统一支持：

- 数据集：`ogbn-arxiv`、`cora`、`pubmed`、`amazon-photo`
- 特征：`bow`、`plm`
- 模型：`mlp`、`gcn`、`sage`、`gat`
- 指标：`Accuracy`、`Macro-F1`
- 重复实验：支持多次运行并自动汇总均值与标准差

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

`torch-geometric` 在 Windows 环境下通常需要和本地 `torch` 版本匹配安装。如果直接安装失败，先按你的 CUDA / CPU 环境安装 `torch`，再安装 `torch-geometric`。

## 2. 运行实验

```bash
python main.py
```

默认会尝试跑完整表格。

可选参数示例：

```bash
python main.py --datasets cora pubmed --feature-types bow --models gcn gat
```

多次重复实验示例：

```bash
python main.py --runs 5 --seed 42
```

## 3. BoW 特征

`bow` 默认直接使用数据集自带的节点特征矩阵 `data.x`：

- `cora` / `pubmed` / `amazon-photo`：通常就是词袋或稀疏文本特征
- `ogbn-arxiv`：默认使用 OGB 自带特征

## 4. PLM 特征

PLM 特征优先级如下：

1. `data/manual_features/{dataset}_plm.pt`
2. `data/manual_features/{dataset}_plm.npy`
3. `data/texts/{dataset}.jsonl`
4. `data/texts/{dataset}.csv`
5. `data/texts/{dataset}.txt`

如果提供的是原始文本，程序会用 `--plm-model` 指定的 Hugging Face 模型编码，并缓存到：

```text
data/feature_cache/
```

### 文本文件格式

`jsonl`：

```json
{"text": "first node text"}
{"text": "second node text"}
```

`csv`：

```csv
text
first node text
second node text
```

要求文本条数和节点数完全一致，按节点编号顺序对齐。

## 5. 输出结果

运行后会生成：

- `outputs/results_raw.csv`
- `outputs/results_long.csv`
- `outputs/results_wide.csv`
- `outputs/run_config.txt`

说明：

- `results_raw.csv`：每次运行一行，适合查训练波动
- `results_long.csv`：按 `feature_type + model + dataset` 汇总后的均值/标准差
- `results_wide.csv`：宽表格式，最适合直接抄到论文表格
