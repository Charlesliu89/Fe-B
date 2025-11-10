#!/usr/bin/env python3
"""
Interactive enthalpy of mixing calculator for a binary mixture using

    ΔH_mix = 4 · ( Σ_{k=0}^3 Ω_k · (c_A - c_B)^k ) · c_A · c_B

where c_A and c_B are mole fractions of the two components and Ω_k are
interaction coefficients (kJ/mol). The script guides the user through
collecting mole fractions and Ω_k values, then reports the resulting
ΔH_mix along with each polynomial contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# 常量：多项式最高阶数
POLY_ORDER = 3


@dataclass
class BinaryMixture:
    """保存二元体系的名称、摩尔分数与相互作用系数。"""

    names: List[str]
    mole_fractions: List[float]  # [c_A, c_B]
    omegas: List[float]  # Ω_0 ... Ω_3

    def enthalpy_of_mixing(self) -> float:
        """计算 ΔH_mix (kJ/mol)。"""
        c_a, c_b = self.mole_fractions
        delta = c_a - c_b
        polynomial = sum(omega * (delta ** k) for k, omega in enumerate(self.omegas))
        return 4.0 * polynomial * c_a * c_b

    def polynomial_terms(self) -> List[float]:
        """返回每一阶 Ω_k · (c_A - c_B)^k 的值。"""
        c_a, c_b = self.mole_fractions
        delta = c_a - c_b
        return [omega * (delta ** k) for k, omega in enumerate(self.omegas)]


def prompt_float(prompt: str, default: float | None = None) -> float:
    """读取浮点数，支持默认值。"""
    while True:
        raw = input(prompt).strip()
        if not raw:
            if default is not None:
                return default
            print("请输入一个数字。")
            continue
        try:
            return float(raw)
        except ValueError:
            print("无法解析该数值，请重新输入。")


def gather_binary_components() -> List[str]:
    """获取两个组分名称。"""
    names: List[str] = []
    for idx in range(2):
        name = input(f"组分 {idx + 1} 名称 (留空默认为 Component_{idx + 1}): ").strip()
        if not name:
            name = f"Component_{idx + 1}"
        names.append(name)
    return names


def gather_mole_fractions(names: List[str]) -> List[float]:
    """获取并归一化两个组分的摩尔分数。"""
    fractions: List[float] = []
    for name in names:
        fraction = prompt_float(f"{name} 的摩尔分数(可不归一化): ")
        if fraction < 0:
            raise ValueError("摩尔分数不能为负数。")
        fractions.append(fraction)

    total = sum(fractions)
    # 归一化，确保 c_A + c_B = 1
    if total <= 0:
        raise ValueError("摩尔分数总和必须为正。")
    return [x / total for x in fractions]


def gather_omegas() -> List[float]:
    """读取 Ω_0 ... Ω_3 系数。"""
    omegas: List[float] = []
    for k in range(POLY_ORDER + 1):
        prompt = f"Ω_{k} (kJ/mol，默认 0.0): "
        omegas.append(prompt_float(prompt, default=0.0))
    return omegas


def build_mixture() -> BinaryMixture:
    """通过交互输入构建 BinaryMixture 实例。"""
    names = gather_binary_components()
    mole_fractions = gather_mole_fractions(names)
    omegas = gather_omegas()
    return BinaryMixture(names, mole_fractions, omegas)


def display_results(mixture: BinaryMixture) -> None:
    """输出计算结果及每一阶贡献。"""
    print("\n=== 计算结果 ===")
    for name, frac in zip(mixture.names, mixture.mole_fractions):
        print(f"{name:<15s} x = {frac:.4f}")

    print("\n多项式各阶贡献 (kJ/mol):")
    for k, term in enumerate(mixture.polynomial_terms()):
        print(f"Ω_{k} · (c_A - c_B)^{k}: {term:>10.5f}")

    print(f"\nΔH_mix (kJ/mol): {mixture.enthalpy_of_mixing():.5f}")


def main() -> None:
    mixture = build_mixture()
    display_results(mixture)


if __name__ == "__main__":
    main()
