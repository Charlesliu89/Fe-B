# 混合焓（ΔHmix）工具

本仓库提供两个 Python 脚本，基于 `Data/Element pair data base matrices.xlsx` 中的 Ω 矩阵，帮助计算与可视化合金的混合焓。

- **enthalpy of mixing.py**：交互式单配方计算器，读取一行元素配比后输出归一化原子分数、各元素对的贡献以及总 ΔH<sub>mix</sub>。
- **batch_enthalpy.py**：批量绘图脚本，利用 Plotly + Kaleido 生成二元曲线与三元等值线 PNG（必要时回退到 HTML）。

## 环境与准备
1. 使用 Python 3 安装依赖：
   ```bash
   python3 -m pip install -r requirements.txt
   ```
2. 确保 Ω 工作簿存在于 `Data/Element pair data base matrices.xlsx`，或在运行时提供其他路径。

> 输出目录、PNG 导出都由脚本自动创建；若 Kaleido 不可用，脚本会自动改存交互式 HTML。

## 交互式计算器（enthalpy of mixing.py）
```bash
python "enthalpy of mixing.py"
```
- 直接回车使用默认 Excel 路径，或输入自定义路径。
- 在一行中输入配比，支持空格、逗号、冒号、等号等分隔符，例如：
  - `Fe 20 Al 30 Ni 50`
  - `Fe=20,Al=30,Ni=50`
- 脚本会自动归一化原子分数，列出所有元素对的 ΔH<sub>mix</sub> 贡献并汇总总值；无效元素符号或缺失数据会被明确提示。

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
3. 四元 ΔH_mix 预览：输入四种元素后，以可旋转的等边四面体展示组成空间与混合焓分布，可选将某一元素含量固定后导出对应的三元投影 PNG/HTML。
4. 自定义组合预览（Plotly 交互式曲线/三角图，可选择导出）。

生成的 PNG 保存在所选输出目录下；若遇到 PNG 导出异常，将在同一路径输出等效的 HTML 以便查看。

## 数据与输出
- `Data` 目录应包含 Ω 矩阵 Excel 与脚本生成的所有图像导出，仓库本身不附带源数据。
- 批量运行或自动模式会自动创建缺失的输出子目录，并在控制台打印跳过的组合与保存路径总结。
