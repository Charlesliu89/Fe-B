# 混合焓（ΔHmix）工具

本仓库基于 `Data/Element pair data base matrices.xlsx` 中的 Ω 矩阵，提供混合焓计算与可视化工具，模块化拆分如下：

- **enthalpy_config.py**：常量配置（字体/颜色条、Plotly 导出参数、采样步长、Ω 表路径等）。
- **enthalpy_core.py**：核心计算与数据处理（元素规范化、Ω 表加载、配比解析、ΔH 计算、采样点生成等）。
- **enthalpy_plot.py**：Plotly 绘图与导出（统一样式的二元/三元/四元图形）。
- **enthalpy of mixing.py**：交互式 / 命令行单配方计算器，读取一行元素配比后输出归一化原子分数、各元素对的贡献以及总 ΔH<sub>mix</sub>。
- **batch_enthalpy.py**：批量绘图入口（批量二元/三元、四元预览与切片、自定义组合），调用上述模块完成计算与绘制。

## 环境与准备
1. 使用 Python 3 安装依赖：
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. 确保 Ω 工作簿存在于 `Data/Element pair data base matrices.xlsx`，或在运行时提供其他路径。

> 输出目录、PNG 导出都由脚本自动创建；若 Kaleido 不可用，脚本会自动改存交互式 HTML。

## 交互式/命令行计算器（enthalpy of mixing.py）
```bash
python "enthalpy of mixing.py" [--excel PATH] [--line "Fe 20 Al 80"] [--list-elements]
```
- `--excel PATH`：指定 Ω 工作簿路径（默认 `Data/Element pair data base matrices.xlsx`）。
- `--line`：一次性计算一行配比，直接输出结果后退出；省略时进入交互模式。
- `--list-elements`：打印工作簿支持的元素列表后退出。
- 输入格式：支持空格、逗号、冒号、等号等分隔符，例如 `Fe 20 Al 30 Ni 50` 或 `Fe=20,Al=30,Ni=50`。
- 输出：归一化原子分数、各元素对的 ΔH<sub>mix</sub> 贡献以及总 ΔH<sub>mix</sub>；无效符号或缺失数据会有明确提示。

## 批量绘图（batch_enthalpy.py）
```bash
python batch_enthalpy.py [options]
```
常用参数：
- `--excel-db PATH`：指定 Ω 工作簿路径，默认复用计算器脚本中的设置。
- `--calculator PATH`：替换用于计算 ΔH<sub>mix</sub> 的脚本（默认同仓库内计算器）。
- `--elements Fe B Ni`：限制组合使用的元素池，默认使用工作簿中所有可用元素。
- `--output-dir Data/plots`：输出根目录；二元、三元、定制结果分别存放于 `binary/`、`ternary/`、`custom/` 子目录，并按 `Fe-Ni.png`、`Fe-Co-Ni.png` 命名。
- `--workers 4`：并行进程数（默认为 CPU 核心数）。
- `--auto-combo Fe,B,Ni --auto-combo Fe,Co`：非交互模式，按提供的元素列表直接生成图并退出。

### 交互式菜单
直接运行脚本并按提示选择：
1. 批量生成所有支持的二元 ΔH<sub>mix</sub> 曲线（0–100%，步长 0.1%）。
2. 批量生成所有支持的三元 ΔH<sub>mix</sub> 等值线图。
3. 四元 ΔH_mix 预览：输入四种元素后，以可旋转的等边四面体展示组成空间与混合焓分布；可在预览后固定某一元素的浓度（预览步长默认 1%，切片重采样步长 0.1%）并导出三元投影 PNG/HTML。
4. 自定义组合预览（Plotly 交互式曲线/三角图/四元预览，可选择导出）。

生成的 PNG 保存在所选输出目录下；若遇到 PNG 导出异常，将在同一路径输出等效的 HTML 以便查看。

## 数据与输出
- `Data` 目录应包含 Ω 矩阵 Excel 与脚本生成的所有图像导出，仓库本身不附带源数据。
- 批量运行或自动模式会自动创建缺失的输出子目录，并在控制台打印跳过的组合与保存路径总结。
