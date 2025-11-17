# 混合焓（ΔHmix）工具

本仓库包含两个用于可视化合金混合焓（ΔH<sub>mix</sub>）的 Python 工具，它们依赖于存放在 `Data/Element pair data base matrices.xlsx` 中的 Ω 矩阵：

- **enthalpy of mixing.py**：交互式命令行计算器。输入一行元素符号和百分比后，脚本会输出归一化原子分数、各元素对的贡献以及该配方的总 ΔH<sub>mix</sub>。
- **batch_enthalpy.py**：批量绘图脚本，可遍历所有二元/三元组合，跳过不受支持的配对，并导出 Matplotlib/Plotly 静态图。

## 环境配置

1. 使用 Python 3 安装依赖：`python3 -m pip install -r requirements.txt`。
2. 确保 Ω 工作簿位于 `Data/Element pair data base matrices.xlsx`（或在提示/通过 `--excel-db` 参数提供其他路径）。

## 运行交互式计算器

```bash
python "enthalpy of mixing.py"
```

- 直接回车使用默认的 Excel 路径，或输入自定义路径。
- 在一行中输入配比，例如 `Fe 20 Al 30 Ni 50` 或 `Fe=20,Al=30,Ni=50`。
- 脚本会报告归一化原子分数、逐对 ΔH 贡献及合计的 ΔH<sub>mix</sub>。

## 批量绘图

```bash
python batch_enthalpy.py [options]
```

常用选项：

- `--elements Fe B Ni`：限制组合时使用的元素池。
- `--output-dir Data/plots`：指定输出目录（生成的二元/三元图分别位于 `binary/` 和 `ternary/` 子目录）。
- `--workers 4`：在遍历大量组合时设置并行进程数。
- `--calculator` / `--excel-db`：指定替代的计算器脚本或 Ω 工作簿路径。

### 非交互自动模式

使用 `--auto-combo` 可跳过文本菜单，直接生成指定元素组合的图。传入以逗号或空格分隔的元素列表；重复该参数即可一次渲染多组组合：

```bash
python batch_enthalpy.py \
  --auto-combo Fe,B,Ni \
  --auto-combo Fe,Co
```

生成的图会保存在所选 `--output-dir`（默认 `Data/plots`）下；不兼容的元素组合会被记录并跳过。

## 数据说明

`Data` 目录应包含两个脚本访问的 Excel 矩阵，以及任何生成的图像导出。仓库本身不包含测试数据。
