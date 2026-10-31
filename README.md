# JM

Joint Moment Integration 将高斯像素区域颜色和原始独立参数响应表示为共同剖面幂与空间单项式的组合，以共享区域矩完成积分。

## 安装

Python 3.11、PyTorch 2.5.1、CUDA 12.1、NumPy 1.26.4、SciPy 1.13.1。

```bash
conda env create -f environment.yml
conda activate jm
pip install -e .
```

CPU 也可执行全部入口。场景参数与投影使用 PyTorch，区域多项式、矩收缩和空间区间计算使用 NumPy。`--device cuda` 将场景张量和投影置于 CUDA；区域积分在 CPU 执行。运行时间取决于图像尺寸、有效图层数量和自适应细分。

## 数学接口

```python
from jm import Settings, integrate

color, response, appearance, terminal = integrate(
    mu, precision, opacity, colors, background,
    rect=[-0.5, -0.5, 0.5, 0.5], settings=Settings()
)
```

输入按前向后深度顺序排列：`mu` 为 `[N,2]`，`precision` 为 `[N,2,2]` 正定矩阵，`opacity` 为 `(0,1]` 中的 `[N]` 数组，`colors` 为 `[N,C]`，`background` 为 `[C]`。矩形使用 `[xmin,ymin,xmax,ymax]`。颜色与背景通道取值在 `[0,1]`。有量纲的线性通道先归一化再恢复尺度。

`response[i,b,c]` 是第 i 个独立高斯的原始 log-response 取空间基函数 b 时，积分后的通道响应。基函数顺序为 `1, xi, v, xi², xi*v, v²`，局部坐标由区域中心与半宽定义。任意二次方向系数 `p[N,6]` 的几何响应为：

```python
direction = (response * p[:, :, None]).sum(axis=(0, 1))
direction += appearance @ directional_colors
direction += terminal * directional_background
```

`Regional` 为独立原参数响应提供 PyTorch 反向接口。其反向值是 JM 响应公式的收缩，训练方向由该响应与图像残差共同形成。JM 颜色近似与响应近似分别构造。

## 计算流程

1. 使用不透明度权重形成共同中心和精度矩阵，通过矩形上的二次型极值约束剖面幅度。
2. 反向合成颜色 B 与一阶残差 R，保留完整剖面幂次数。
3. 使用前缀透射率及其残差变化，构造各原参数的共享反向收缩。
4. 只计算不同的 `g^k * xi^a * v^b` 区域矩，空间总次数至多为四。
5. 二次残差极值控制剖面细分；双倍 Gauss–Legendre 阶数控制积分精度；达到深度上限或图层路由阈值时使用 Direct。
6. 每个候选状态重新投影与排序。原始高斯合成的空间区间包络给出目标区间，仅在候选上界小于当前下界时接受更新。

`direct_layers` 控制层数对应的计算成本路由。设为 `0` 可运行 Direct 积分；增大该值可扩大共同矩分支的适用层数。数值积分若在最大阶数内未达到要求，会中止并报告待调整参数。

## 数据

每个建筑单元使用独立目录：

```text
data/T01/
  scene.json
  init.npz
  images/
```

`scene.json`：

```json
{
  "initialization": "init.npz",
  "views": [
    {
      "image": "images/000001.png",
      "K": [[1000, 0, 749.5], [0, 1000, 499.5], [0, 0, 1]],
      "w2c": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
      "split": "train"
    }
  ]
}
```

相机使用 OpenCV 轴约定：x 向右、y 向下、z 向前；`w2c` 从世界坐标变换到相机坐标，`K` 使用像素坐标。图像像素中心为整数坐标。场景长度单位为米。`split` 为 `train` 或 `test`；训练目标和融合使用训练视图。

`init.npz` 包含 `xyz[N,3]` 和可选的 `rgb[N,3]`、`scale[N,3]`、`opacity[N]`。RGB 为 `[0,1]`，scale 是三个轴的正标准差。优化状态将协方差表示为带指数对角线的 Cholesky 因子，将颜色和不透明度表示为 sigmoid 参数。

ETH3D 与 GauU-Scene 的已校准建筑单元可通过 COLMAP 去畸变后的文本模型转换：

```bash
python tools/prepare_colmap.py --model /data/T01/sparse_txt --images /data/T01/images --out data/T01 --train-views 12 --test-views 4 --width 1500 --height 1000
```

转换工具读取 `cameras.txt`、`images.txt`、`points3D.txt`，保存确定性视图划分、缩放后的内参和点云初始化。`--max-points` 指定初始化点数上限；`--seed` 控制视图与点采样。

## 运行

```bash
jm demo --out data/demo
jm train --data data/demo --config configs/demo.json --out runs/demo
jm render --data data/demo --checkpoint runs/demo/scene.npz --config configs/demo.json --out runs/demo/render

jm train --data data/T01 --config configs/T.json --out runs/T01
jm train --data data/G01 --config configs/G.json --out runs/G01
jm extract --data data/T01 --checkpoint runs/T01/scene.npz --config configs/T.json --out runs/T01/surface
```

`scene.npz` 保存场景参数；`training.csv` 记录尝试次数、接受更新数、完整经过时间、目标区间和步长；`run.json` 保存汇总。无法判定下降时，先收紧区间，再减小步长；达到配置的搜索次数后保存当前状态并结束。

渲染输出 RGB、中心深度和覆盖率。提取时屏蔽低覆盖区域与多视图不一致深度，再进行 TSDF 融合和等值面提取，结果为 `surface.ply`。TSDF 默认容许两千万体素；较大建筑区域可在 `surface.fuse` 中设置 `max_voxels`。

## 配置

| 参数 | T | G |
| --- | --- | --- |
| 接受更新数 | 30000 | 30000 |
| 几何正则权重 | 0.1 | 0.1 |
| 法向项系数 | 1 | 1 |
| 颜色局部容差 | 1e-4 | 1e-4 |
| 单位 log-response 局部容差 | 1e-3 | 1e-3 |
| 最大细分深度 | 4 | 4 |
| 覆盖率门槛 | 0.01 | 0.01 |
| 体素尺寸 | 0.02 m | 0.08 m |
| 截断距离 | 4 体素 | 4 体素 |
| F-score 距离门槛 | 0.10 m | 0.40 m |

`configs/T_60min.json`、`G_60min.json`、`T_90min.json`、`G_90min.json` 提供固定时间设置。时间从训练初始化后开始，包括目标计算、响应收缩、候选检查和周期保存。达到时间上限后在当前更新结束时退出。

## 评估与验证

将预测与参考表面按 T 的 0.02 m 或 G 的 0.08 m 间距采样为 `[M,3]` NPY 点集：

```bash
jm metrics --prediction prediction.npy --reference reference.npy --threshold 0.1 --out metrics.json
python -m pytest tests -q
```

指标包括 accuracy、completeness、CD-L1、precision、recall 和 F-score。`jm.metrics.paired_interval` 接收建筑 × 重复运行数组，先对重复运行求均值，再对配对建筑差值计算 Student t 置信区间。

验证涵盖共同剖面精确情形、原参数有限差分、细分坐标转换、空间区间包络、有限高斯可见性反例、四层联合展开及二阶误差缩放、PyTorch 原参数响应回传。

## 目录

```text
jm/integrator.py    共同剖面、完整多项式阶数、共享矩与自适应路由
jm/polynomial.py    空间多项式和矩形二次型极值
jm/enclosure.py     原始合成空间区间与目标包络
jm/render.py        透视投影、原参数回传、图像和深度通道
jm/train.py         几何正则、候选检查与训练记录
jm/data.py          相机数据与场景参数读写
jm/surface.py       多视图深度检查和 TSDF 表面提取
jm/metrics.py       表面指标与配对统计
jm/cli.py           命令行入口
tools/prepare_colmap.py  校准数据转换
configs/           更新数、时间预算和建筑尺度配置
tests/             数值与响应验证
```
