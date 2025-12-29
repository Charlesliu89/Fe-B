# Fe-B ΔH<sub>mix</sub> Toolkit / Fe-B 混合焓工具集

Python 工具集合，用于基于 `Data/Element pair data base matrices.xlsx` 中的 Omega 互作用矩阵计算与可视化合金混合焓。  
This toolkit reads Omega matrices from `Data/Element pair data base matrices.xlsx` to compute and plot alloy mixing enthalpy.

功能 / Features:
- 交互式单点计算器，输出成分归一化、成对贡献与总 ΔH<sub>mix</sub>。  
  Interactive single-composition calculator with normalized fractions, pair contributions, and total ΔH<sub>mix</sub>.
- 批处理绘图器，可批量导出二元曲线、三元等高线、四元预览/切片及等摩尔五元排名。  
  Batch plotter for binary curves, ternary contours, quaternary previews/slices, and equimolar quinary rankings.
- 零混合焓设计：输入 2–5 种元素，通过优化搜索 ΔH_mix 接近 0 的成分组合。  
  Zero-enthalpy design: optimize 2–5 element compositions with ΔH_mix near 0.
- 专用 Fe-B-Cr 绘图，支持将相界对齐采样网格后叠加。  
  Dedicated Fe-B-Cr plot with optional snapped phase-boundary overlays.

核心模块 / Core modules:
- `enthalpy_core.py`: 数据加载与计算。 / Data loading and math utilities.
- `enthalpy_plot.py`: Plotly 图形构建与导出。 / Plotly figure builders and export helpers.
- `enthalpy_config.py`: 共享常量（字体、步长、路径等）。 / Shared constants (fonts, steps, paths).

## 安装 / Installation
1) Python 3.10+  
2) （可选）创建虚拟环境 / Optional venv  
3) 安装依赖 / Install deps:
```bash
python -m pip install -r requirements.txt
```
> PNG 导出依赖 Kaleido；若失败将自动回退为 HTML。  
> PNG export uses Kaleido; falls back to HTML if PNG export fails.

## 数据准备 / Data preparation
- 默认读取 `Data/Element pair data base matrices.xlsx`。  
  Workbook is expected at `Data/Element pair data base matrices.xlsx`.
- 数据文件与生成图片不提交到仓库，请在本地准备并保留。  
  Data files and generated images are kept locally (not tracked in git).
- 确认工作表 `U0`、`U1`、`U2`、`U3` 已按原子序填充配对系数；如有更新，请同步 Excel。  
  Ensure sheets `U0`–`U3` contain pair coefficients ordered by atomic number; update Excel when data change.

## 脚本与用法 / Scripts and usage

### 1) `single_enthalpy_cli.py`
交互或一次性计算 2–5 元成分。 / Interactive or one-shot calculator for 2–5 element compositions.
```bash
python single_enthalpy_cli.py [--excel PATH] [--line "Fe 20 Al 80"] [--list-elements]
```
- `--excel PATH`：自定义 Omega 工作簿。 / Override workbook.
- `--line`：提供单行成分，自动归一化后计算。 / One-shot composition line.
- `--list-elements`：列出可用元素后退出。 / List supported elements.
未加 `--line` 时进入 REPL，可输入 `Fe 20 Al 30 Ni 50` 或 `Fe=20,Al=30,Ni=50`。  
Without `--line`, enter REPL; examples: `Fe 20 Al 30 Ni 50`, `Fe=20,Al=30,Ni=50`.

### 2) `batch_enthalpy_plots.py`
批量绘图；无参数运行进入菜单。 / Batch plot generator; run without args for menu.
```bash
python batch_enthalpy_plots.py
```
菜单 / Menu:
1. 二元曲线 → `Data/plots/binary`  
   Binary curves
2. 三元等高线 → `Data/plots/ternary`  
   Ternary contour plots
3. 四元四面体预览与切片 → `Data/plots/quaternary`  
   Quaternary preview + ternary slices
4. 自定义 2–4 元组合 → `Data/plots/custom`  
   Custom plots (2–4 elements)
5. 等摩尔五元排名，可导出 Excel → `Data/plots/quinary`  
   Equimolar quinary ranking with optional Excel export
6. 零混合焓成分设计（2–5 元素）  
   Zero-enthalpy composition design (2–5 elements)

常用参数 / Useful flags:
- `--calculator PATH`：指定计算核心（默认 `single_enthalpy_cli.py`）。 / Alternate calculator module.
- `--excel-db PATH`：显式指定工作簿。 / Explicit workbook path.
- `--elements Fe B Ni ...`：限制元素池。 / Restrict element pool.
- `--output-dir Data/plots`：修改输出根目录，子目录自动创建。 / Change output root.
- `--auto-combo Fe,B,Ni --auto-combo Fe,Co`：跳过菜单直接渲染指定组合（结果存入 `custom/`）。 / Render listed combos directly.
- `--workers 8`：并行进程数（默认 CPU 数）。 / Worker processes.
- `--chunk-size 50`，`--chunk-auto-continue`：控制批大小与是否自动继续，便于长任务或 CI。 / Batch size and unattended mode.
- `--list-elements`：列出支持元素后退出。 / List elements then exit.
Kaleido 导出失败时会自动保存 `.html` 交互文件。  
If Kaleido PNG export fails, `.html` fallbacks are saved.

### 3) `fe_b_cr_phase_overlay.py`
生成 Fe-B-Cr 三元图，可选对齐网格的相界叠加。 / Fe-B-Cr ternary plot with optional snapped phase boundaries.
```bash
python fe_b_cr_phase_overlay.py \
    [--excel Data/Element pair data base matrices.xlsx] \
    [--step 0.001] \
    [--boundary-csv Data/FeBCr phase diagram line.csv] \
    [--output Data/FeBCr_phase_boundaries.png] \
    [--show] [--no-boundaries]
```
- `--step`：采样步长（默认 0.001 = 0.1%）。 / Sampling resolution.
- `--boundary-csv`：相界 CSV，列 a/b/c 会被清洗并对齐网格。 / Boundary CSV (a/b/c columns), cleaned and snapped.
- `--output`：PNG 路径（自动创建目录），可配合 `--show` 同时预览。 / PNG path; add `--show` for preview.
- `--no-boundaries`：禁用相界叠加。 / Disable boundary overlay.

## 提示 / Tips
- 首次运行请核对元素列表与 Excel 完全一致，缺项通常是表头/行格式问题。  
  Verify listed elements match Excel; mismatches usually indicate header/row issues.
- 批处理时用 `--elements` 收缩元素池，可减少因缺数据而跳过的组合。  
  Narrow the element pool with `--elements` to avoid unsupported combos.
- 四元预览支持固定单元素比例（如 `Fe=25`），可导出三元 PNG/HTML。  
  Quaternary preview supports slices (e.g., `Fe=25`) and exports ternary PNG/HTML.
- 等摩尔五元排名按 ΔH<sub>mix</sub> 正负分组，支持导出 Excel 便于后续分析。  
  Equimolar quinary ranking groups by sign of ΔH<sub>mix</sub> and can export to Excel.

---

扩展新脚本或配色时，请将通用常量集中在 `enthalpy_config.py`，以保持各模块一致。  
When extending workflows (new calculators, colour maps, etc.), keep shared constants in `enthalpy_config.py` for consistency.
