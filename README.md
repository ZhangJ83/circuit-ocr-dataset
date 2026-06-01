# Circuit Schematic OCR — 基于 PaddleOCR-VL 的电路原理图理解与网表提取

> **PaddleOCR 全球衍生模型挑战赛** | Hackathon 10th | GitHub Issue #17858

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v2.7%2B-orange.svg)]()

## 项目亮点

### 🔬 核心创新

1. **零人工标注数据管线** — 所有标注从 KiCad 源文件程序化提取，零人工成本
2. **真实退化图像鲁棒性** — 5 种退化类型（纸张老化、扫描噪声、透视变形、手写叠加、低分辨率），所有现有工作未覆盖
3. **数字/混合信号电路** — 覆盖 MCU、FPGA、PMIC 等复杂电路，超越 Image2Net 的纯模拟电路
4. **多层次结构化输出** — 文字→元件→连接关系→功能语义，端到端理解
5. **SPICE 自动仿真验证** — 闭环验证：图片→OCR→网表→仿真→修正

### 📊 评分维度覆盖

| 评分维度 | 分值 | 我们的策略 |
|---------|------|-----------|
| 评估集质量 | 20 | 高质量评估集，覆盖复杂度/退化/特殊字符 |
| 场景稀缺性 | 15 | 电路原理图 OCR，排行榜唯一 |
| 任务复杂度 | 15 | 多层次结构化输出 + 功能语义理解 |
| 训练数据集构建科学性 | 20 | KiCad 源文件驱动 + 合成数据 + 退化增强 |
| 微调策略与创新 | 10 | PaddleOCR-VL 端到端微调 + SPICE 验证闭环 |
| 技术文档与开源贡献 | 20 | 完整文档 + 可复现 + 社区价值 |

## 快速开始

### 环境配置

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 安装 KiCad 8 (含 kicad-cli)
# https://www.kicad.org/download/

# 3. 安装 PaddlePaddle & PaddleOCR
pip install paddlepaddle-gpu paddleocr

# 4. (可选) 安装 ngspice 用于仿真验证
# https://ngspice.sourceforge.io/
```

### 一键运行 Demo

```bash
python scripts/demo.py --generate
```

### 完整数据集构建

```bash
# Step 1: 从 GitHub 采集 KiCad 项目
python scripts/collect_data.py --max-repos 200 --github-token YOUR_TOKEN

# Step 2: 构建数据集（解析+渲染+标注+合成+退化）
python scripts/build_dataset.py --project-dir .

# 仅使用合成数据（无需 GitHub token）
python scripts/build_dataset.py --skip-scraping --synthetic-count 500
```

### 模型训练

```bash
# 检测模型训练
python scripts/train.py --task det

# 识别模型训练
python scripts/train.py --task rec

# 全部训练
python scripts/train.py --task all
```

### 推理

```bash
# 单张图片推理
python scripts/infer.py --image path/to/schematic.png --verify --save-netlist

# 批量推理
python scripts/infer.py --image-dir path/to/images/ --output-dir results/
```

### 评估

```bash
python scripts/evaluate.py --eval-dir data/eval --output-dir output/eval
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Circuit Schematic OCR                      │
├───────────────┬───────────────┬───────────────┬─────────────┤
│  Data Pipeline │  Model Layer  │  Inference    │ Verification│
├───────────────┼───────────────┼───────────────┼─────────────┤
│ GitHub Scraper│ PP-OCRv4 Det  │ Predictor     │ SPICE Verify│
│ KiCad Parser  │ PP-OCRv4 Rec  │ PostProcessor │ AutoCorrect │
│ Renderer      │ PaddleOCR-VL  │ NetlistExtract│             │
│ Annotation Gen│               │               │             │
│ Synth Generator│              │               │             │
│ Degradation   │               │               │             │
└───────────────┴───────────────┴───────────────┴─────────────┘
```

### 数据管线

```
KiCad Project (.kicad_sch)
    │
    ├──→ KiCadParser ──→ Components, Wires, Labels, Nets
    │
    ├──→ SchematicRenderer ──→ PNG Image (300 DPI)
    │
    ├──→ AnnotationGenerator ──→ PaddleOCR Format Labels
    │
    └──→ SchematicDegradation ──→ 5 Degraded Variants

Synthetic Generator
    │
    ├──→ Random Circuit Spec ──→ .kicad_sch File
    │
    └──→ PIL Rendering ──→ PNG Image
```

### 退化增强管线（差异化 B）

| 退化类型 | 模拟场景 | 实现方式 |
|---------|---------|---------|
| paper_aging | 纸张老化：泛黄、斑点 | 颜色偏移 + 随机斑点 + 暗角 |
| scan_noise | 扫描噪点 | 高斯噪声 + 水平扫描条纹 |
| perspective_distortion | 拍照透视变形 | 随机透视变换 + 旋转 |
| handwriting_overlay | 叠加手写标注 | 随机手写文字/符号/下划线 |
| low_resolution | 低分辨率扫描 | 下采样 + 模糊 + 上采样 |

### 输出格式

```json
{
  "texts": [
    {"text": "R1", "bbox": [[100,200],[150,200],[150,220],[100,220]], "category": "reference"},
    {"text": "10k", "bbox": [[100,230],[150,230],[150,250],[100,250]], "category": "value"},
    {"text": "VCC", "bbox": [[200,100],[250,100],[250,115],[200,115]], "category": "net_label"}
  ],
  "components": [
    {"ref": "R1", "value": "10k", "type": "Resistor", "bbox": [[100,200],[150,230]]},
    {"ref": "C1", "value": "100nF", "type": "Capacitor", "bbox": [[300,200],[350,230]]}
  ],
  "nets": {
    "VCC": ["R1:1", "C1:1"],
    "GND": ["R1:2", "C1:2"]
  },
  "spice_netlist": "* Auto-generated\nR1 VCC NET_1 10k\nC1 NET_1 GND 100nF\n.end"
}
```

## 数据集

### 数据来源

| 来源 | 数量 | 说明 |
|------|------|------|
| GitHub 开源 KiCad 项目 | 200+ 仓库 | MCU/传感器/电源/通信接口 |
| 合成电路 | 300+ 张 | 随机拓扑，覆盖模拟/数字/混合/电源 |
| 退化增强 | 每张 5 变体 | 真实世界退化模拟 |

### 电路类型覆盖

| 类型 | 元件数 | 典型电路 |
|------|--------|---------|
| 模拟 | 5-20 | 运放、滤波器、电源 |
| 数字 | 10-50 | MCU 最小系统、逻辑门、存储器 |
| 混合信号 | 20-100 | 传感器接口、ADC/DAC、PMIC |
| 电源 | 10-30 | LDO、Buck/Boost、电池管理 |

### 特殊字符支持

支持电子学特殊字符：`Ω`, `μ`, `±`, `°`, `℃`, `∞`, `α`, `β`, `γ`, `π`, `σ`, `φ`

## 评估指标

### 四级评估体系

| 级别 | 指标 | 说明 |
|------|------|------|
| L1 文字检测 | Precision, Recall, F1 | IoU > 0.5 |
| L2 文字识别 | Char Accuracy, Word Accuracy | 字符/词级准确率 |
| L3 元件检测 | mAP@0.5, Type Accuracy | 检测+分类 |
| L4 网表提取 | Connection Accuracy, Exact Match | 完整网表匹配 |

## 竞争优势

### vs. Image2Net (最大竞争对手)

| 维度 | Image2Net | 我们的方案 |
|------|-----------|-----------|
| 电路类型 | 纯模拟 (5-10 元件) | 数字/混合信号 (20-100 元件) |
| 输入 | 干净渲染图 | 真实退化图像 |
| 方法 | 混合框架 (CV + 传统) | PaddleOCR-VL 端到端微调 |
| 输出 | 网表 | 多层次结构化 + 功能语义 |
| 验证 | 无 | SPICE 自动仿真验证 |
| 数据标注 | 手动 | 全自动程序化提取 |

### vs. 其他工作

| 工作 | 未覆盖 | 我们覆盖 |
|------|--------|---------|
| CircuitVision | 真实退化图像 | ✅ 5 种退化类型 |
| Circuitry.ai | 数字/混合信号电路 | ✅ MCU/FPGA/PMIC |
| AMSnet-q | 功能语义理解 | ✅ 功能块识别 |
| DocEDA | SPICE 验证闭环 | ✅ 自动仿真验证 |

## 项目结构

```
circuit_ocr/
├── README.md                         # 本文档
├── requirements.txt                  # Python 依赖
├── LICENSE                           # Apache 2.0
├── configs/                          # 训练配置
│   ├── det_ppocrv4.yml              # PP-OCRv4 检测配置
│   └── rec_ppocrv4.yml              # PP-OCRv4 识别配置
├── src/                              # 核心代码
│   ├── data_pipeline/               # 数据管线 (7 个模块)
│   │   ├── github_scraper.py        # GitHub 数据采集
│   │   ├── kicad_parser.py          # KiCad 文件解析
│   │   ├── renderer.py              # SVG/PNG 渲染
│   │   ├── annotation_generator.py  # 标注生成
│   │   ├── synthetic_generator.py   # 合成数据生成
│   │   ├── degradation.py           # 退化增强
│   │   └── dataset_builder.py       # 数据集构建总控
│   ├── model/                       # 模型训练
│   │   ├── train_det.py             # 检测模型训练
│   │   ├── train_rec.py             # 识别模型训练
│   │   ├── finetune_vl.py           # VLM 微调
│   │   └── export_model.py          # 模型导出
│   ├── inference/                   # 推理管线
│   │   ├── predictor.py             # 推理预测器
│   │   ├── post_processor.py        # 后处理
│   │   └── netlist_extractor.py     # 网表提取
│   ├── evaluation/                  # 评估系统
│   │   ├── metrics.py               # 评估指标
│   │   ├── evaluator.py             # 评估器
│   │   └── report_generator.py      # 报告生成
│   └── verification/                # SPICE 验证
│       ├── spice_verifier.py        # SPICE 仿真验证
│       └── auto_corrector.py        # 自动修正
├── scripts/                         # 运行脚本
│   ├── collect_data.py              # 数据采集
│   ├── build_dataset.py             # 数据集构建
│   ├── train.py                     # 训练入口
│   ├── evaluate.py                  # 评估入口
│   ├── infer.py                     # 推理入口
│   └── demo.py                      # 演示
├── tests/                           # 单元测试
├── data/                            # 数据目录
├── output/                          # 模型输出
└── docs/                            # 文档
```

## 引用

```bibtex
@misc{circuit_ocr_2026,
  title={Circuit Schematic OCR: PaddleOCR-VL Based Circuit Understanding and Netlist Extraction},
  author={CircuitOCR Team},
  year={2026},
  howpublished={PaddleOCR Global Derivative Model Challenge}
}
```

## 致谢

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — 基础 OCR 框架
- [KiCad](https://www.kicad.org/) — EDA 工具和文件格式
- [kiutils](https://github.com/mvnmgrx/kiutils) — KiCad 文件解析库
- [ngspice](https://ngspice.sourceforge.io/) — SPICE 仿真器

## License

Apache License 2.0
