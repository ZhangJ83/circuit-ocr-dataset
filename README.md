# CircuitOCR — 电路原理图合成数据集

> 面向电路原理图 OCR 的大规模程序化合成数据集，含标注与退化增强

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()

## 简介

通过 Python 脚本程序化生成电路原理图及其精确标注，无需人工参与。每条样本包含原理图渲染图像与对应的元件编号、参数值、网络标签及结构化网表。

| 属性 | 说明 |
|------|------|
| 样本量 | ~14,000 张 |
| 标注方式 | 程序化自动生成，100% 精确 |
| 电路类型 | 模拟 / 数字 / 混合信号 / 电源 |
| 退化增强 | 5 种真实场景退化模拟 |

## 标注内容

每张原理图附带的标注包含：元件编号（如 R1、C1）、参数值（如 10k、100nF）、网络标签（如 VCC、GND）、元件间连接关系及 Spice 网表。

## 退化增强

| 类型 | 模拟场景 |
|------|---------|
| paper_aging | 纸张老化：泛黄、斑点 |
| scan_noise | 扫描噪点与条纹 |
| perspective_distortion | 拍照透视变形 |
| handwriting_overlay | 叠加手写标注 |
| low_resolution | 低分辨率扫描 |

## 快速开始

```bash
pip install -r requirements.txt
python scripts/build_dataset.py --synthetic-count 500
```

## 引用

```bibtex
@misc{zhang2026circuitocr,
  title={A Synthetic Dataset for Circuit Schematic OCR and Netlist Extraction},
  author={Jianning Zhang and Yifei Chen},
  year={2026},
  url={https://github.com/ZhangJ83/circuit-ocr-dataset},
}
```

## License

Apache License 2.0
