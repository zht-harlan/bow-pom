# BoW / PLM Graph Experiments

用于完成论文实验中的节点分类对比，统一支持：

- 数据集：`ogbn-arxiv`、`cora`、`pubmed`、`amazon-photo`
- 特征：`bow`、`plm`
- 模型：`mlp`、`gcn`、`sage`、`gat`
- 指标：`Accuracy`、`Macro-F1`
- 重复实验：支持多次运行并统计均值与标准差

## 1. 安装依赖

建议先按服务器 CUDA 环境安装 `torch` 和 `torch_geometric`，再安装其余依赖。

## 2. 运行主实验

完整跑 BoW：

```bash
python main.py --datasets ogbn-arxiv cora pubmed amazon-photo --feature-types bow --models mlp gcn sage gat --runs 5 --seed 42
```

完整跑 PLM：

```bash
python main.py --datasets ogbn-arxiv cora pubmed amazon-photo --feature-types plm --models mlp gcn sage gat --runs 5 --seed 42
```

## 3. BoW 特征

`bow` 默认直接使用数据集自带的 `data.x` 作为节点特征。

## 4. PLM 特征

程序读取 `plm` 特征的优先级如下：

1. `data/manual_features/{dataset}_plm.pt`
2. `data/manual_features/{dataset}_plm.npy`
3. `data/texts/{dataset}.jsonl`
4. `data/texts/{dataset}.csv`
5. `data/texts/{dataset}.txt`

如果放的是原始文本，程序会自动编码并缓存到 `data/feature_cache/`。

### 文本格式

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

要求文本条数和节点数完全一致，并且顺序与节点编号一致。

### 离线生成 PLM 特征

先准备文本文件。你可以手工放入，也可以直接自动生成：

```bash
python prepare_texts.py --datasets ogbn-arxiv cora pubmed amazon-photo --top-k 128
```

这条命令会：

- `ogbn-arxiv`：优先尝试读取原始 `title + abstract`
- 其他数据集：从节点特征自动构造伪文本

生成后文本位于：

- `data/texts/ogbn-arxiv.jsonl`
- `data/texts/cora.jsonl`
- `data/texts/pubmed.jsonl`
- `data/texts/amazon-photo.jsonl`

然后执行 PLM 特征编码：

```bash
python generate_plm_features.py --datasets ogbn-arxiv cora pubmed amazon-photo --plm-model sentence-transformers/all-MiniLM-L6-v2 --batch-size 32
```

生成后的文件会保存到：

```text
data/manual_features/
```

例如：

```text
data/manual_features/ogbn-arxiv_plm.pt
```

之后再运行主实验：

```bash
python main.py --datasets ogbn-arxiv cora pubmed amazon-photo --feature-types plm --models mlp gcn sage gat --runs 5 --seed 42
```

## 5. 输出结果

运行后会生成：

- `outputs/results_raw.csv`
- `outputs/results_long.csv`
- `outputs/results_wide.csv`
- `outputs/run_config.txt`

说明：

- `results_raw.csv`：每次运行一行
- `results_long.csv`：汇总均值和标准差
- `results_wide.csv`：适合直接整理到论文表格
