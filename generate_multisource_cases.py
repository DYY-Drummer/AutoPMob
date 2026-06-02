"""文献横断（multi-source）訓練ケースを手作りで生成し、DOF=0 を検証する.

目的：
  既存の訓練ケースの 97.9% が単一ソース完結という偏りを解消するため、
  Claude が手作業で物理的に意味のあるマルチソース（2-3 ソース横断）ケースを
  作成し、自由度ゼロ条件を機械的に検証して JSON 出力する。

出力：multisource_cases.json
  追加用の new ケースリスト。training_cases.json にマージするのは別ステップ。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Set

ROOT = Path(__file__).parent
EQUATIONS_JSON = ROOT / "unified_equations.json"
OUT_JSON = ROOT / "multisource_cases.json"


# ============================================================
# マルチソース・ケースのテンプレート
# 各ケースは physically motivated な「文献横断モデル」
# DOF=0 を満たすように入出力を厳密に設計
# ============================================================

TEMPLATES = [
    # ====== Theme 1: CSTR + 熱移動 (ChemEng + Heat Transfer) ======
    {
        "case_id": "ms_cstr_heat_001",
        "context": "Steady-state design of a CSTR that requires the residence time from the mass balance and the rate of heat removal from the cooling jacket, given the reaction kinetics and heat transfer characteristics.",
        "equations": [
            "handbook_chemeng__eq_3",          # τ = (C_A0 - C_A) / (-r_A)
            "handbook_heat_transfer__eq_5",    # q = U A ΔT_overall
        ],
        "inputs": ["C_{A0}", "C_A", "r_A", "U", "A", "\\Delta T_{overall}"],
        "outputs": ["\\tau", "q"],
    },
    {
        "case_id": "ms_cstr_heat_002",
        "context": "Non-isothermal CSTR design: compute the residence time and the convective heat removal at the wall, given the kinetics and convective heat transfer characteristics.",
        "equations": [
            "handbook_chemeng__eq_3",
            "handbook_heat_transfer__eq_3",    # q_conv = h A_s (T_s - T_∞)
        ],
        "inputs": ["C_{A0}", "C_A", "r_A", "h", "A_s", "T_s", "T_\\infty"],
        "outputs": ["\\tau", "q_{conv}"],
    },
    {
        "case_id": "ms_cstr_heat_radiation_001",
        "context": "For a high-temperature reactor where radiation heat loss is significant, calculate the residence time and the radiative heat flux from the reactor wall, given the surface emissivity and surface temperatures.",
        "equations": [
            "handbook_chemeng__eq_3",
            "handbook_heat_transfer__eq_4",    # q_rad = ε σ A_s (T_s^4 - T_surr^4)
        ],
        "inputs": ["C_{A0}", "C_A", "r_A", "\\epsilon", "\\sigma", "A_s", "T_s", "T_{surr}"],
        "outputs": ["\\tau", "q_{rad}"],
    },
    # ====== Theme 2: PFR + 反応速度（反応工学 + 化学工学） ======
    {
        "case_id": "ms_pfr_kin_001",
        "context": "Sizing a PFR and an equivalent CSTR in series: determine the PFR volume from the integral design equation and the CSTR residence time, given the feed flow and kinetics.",
        "equations": [
            "handbook_reaction_engineering__eq_5",  # V = F_A0 ∫ dX_A / (-r_A)
            "handbook_chemeng__eq_3",               # τ = (C_A0 - C_A) / (-r_A)
        ],
        "inputs": ["F_{A0}", "X_{A,out}", "-r_A", "C_{A0}", "C_A", "r_A"],
        "outputs": ["V", "\\tau"],
    },
    {
        "case_id": "ms_pfr_batch_001",
        "context": "Compare a batch reactor's conversion dynamics with a PFR's volume requirement, given identical reaction kinetics and feed conditions.",
        "equations": [
            "handbook_reaction_engineering__eq_1",   # N_A0 dX_A/dt = -r_A V
            "handbook_reaction_engineering__eq_5",   # V = F_A0 ∫ dX_A / (-r_A)
        ],
        "inputs": ["N_{A0}", "X_A", "-r_A", "F_{A0}", "X_{A,out}"],
        "outputs": ["V", "t"],
    },
    # ====== Theme 3: 反応器 + プロセス制御 ======
    {
        "case_id": "ms_reactor_pid_001",
        "context": "Compute both the steady-state residence time of a CSTR and the PID controller output for maintaining the desired conversion, given the kinetics and controller parameters.",
        "equations": [
            "handbook_chemeng__eq_3",           # τ = (C_A0 - C_A) / (-r_A)
            "handbook_process_control__eq_1",   # PID
        ],
        # τ は両式で異なる意味（一方は residence time、PID では積分変数）だが、
        # 表記上同一なので DOF 上は 1 個の変数として扱う
        "inputs": ["C_{A0}", "C_A", "r_A", "K_p", "K_i", "K_d", "e(t)", "r(t)", "y(t)", "t"],
        "outputs": ["\\tau", "u(t)"],
    },
    {
        "case_id": "ms_reactor_sensor_001",
        "context": "For a continuous reactor with a measurement sensor, determine the steady-state residence time from the reactor kinetics and the first-order plus dead time (FOPDT) transfer function of the sensor.",
        "equations": [
            "handbook_chemeng__eq_3",           # τ = (C_A0 - C_A) / (-r_A)
            "handbook_process_control__eq_5",   # G_p(s) = K_p e^(-θs) / (τ_p s + 1)
        ],
        "inputs": ["C_{A0}", "C_A", "r_A", "K_p", "\\theta", "\\tau_p", "s"],
        "outputs": ["\\tau", "G_p(s)"],
    },
    # ====== Theme 4: バイオプロセス ======
    {
        "case_id": "ms_bioreactor_001",
        "context": "For a chemostat (continuous bioreactor) at steady state, determine the specific growth rate from Monod kinetics and the substrate consumption residence time analog, given the limiting substrate concentration.",
        "equations": [
            "handbook_bioprocess__eq_1",       # μ = μ_max S / (K_S + S)
            "handbook_chemeng__eq_3",          # τ = (C_A0 - C_A) / (-r_A)
        ],
        "inputs": ["\\mu_{max}", "S", "K_S", "C_{A0}", "C_A", "r_A"],
        "outputs": ["\\mu", "\\tau"],
    },
    # ====== Theme 5: 物質移動 + 反応 ======
    {
        "case_id": "ms_diffusion_reaction_001",
        "context": "In a diffusion-controlled reactor, calculate the molecular diffusion flux at the catalyst surface and the reactor residence time, given the diffusion coefficient, concentration gradient, and reactor kinetics.",
        "equations": [
            "handbook_mass_transfer__eq_1",    # J_Az = -D_AB dC_A/dz
            "handbook_chemeng__eq_3",          # τ = (C_A0 - C_A) / (-r_A)
        ],
        "inputs": ["D_{AB}", "C_A", "z", "C_{A0}", "r_A"],
        "outputs": ["J_{Az}", "\\tau"],
    },
    # ====== Theme 6: Arrhenius (PDF 横断) ======
    {
        "case_id": "ms_arrhenius_001",
        "context": "Compute the temperature-dependent reaction rate constant from the Arrhenius expression and the corresponding reaction rate for a first-order reaction, combining definitions from two process control references.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",  # Arrhenius
            "Process_Control_Designing_Processes_and_Control_Systems_for_Dynamic_Performance.pdf__p0481-0560_eq_14.13_reaction_rate",  # r = k C_A
        ],
        "inputs": ["k_0", "E", "R", "T", "C_A"],
        "outputs": ["k", "r_a"],
    },
    # ====== Theme 7: PFR + Arrhenius ======
    {
        "case_id": "ms_pfr_arrhenius_001",
        "context": "Size a PFR for a first-order Arrhenius reaction: combine the temperature-dependent rate constant from one source with the PFR design equation from another.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",  # k = k_0 exp(-E/RT)
            "handbook_reaction_engineering__eq_4",  # dX_A/dV = -r_A/F_A0
        ],
        "inputs": ["k_0", "E", "R", "T", "X_A", "V", "F_{A0}"],
        "outputs": ["k", "-r_A"],
    },
    # ====== Theme 8: 反応速度のクロス検証 ======
    {
        "case_id": "ms_dual_reaction_001",
        "context": "Use two equivalent expressions for CSTR residence time (one based on concentration difference, another on conversion) to cross-validate the reactor design.",
        "equations": [
            "handbook_chemeng__eq_3",                       # τ = (C_A0 - C_A) / (-r_A)
            "handbook_reaction_engineering__eq_3",          # τ = V/v_0 = C_A0 X_A / (-r_A)
        ],
        # 2 方程式、2 未知数: τ と X_A
        "inputs": ["C_{A0}", "C_A", "r_A", "V", "v_0", "-r_A"],
        "outputs": ["\\tau", "X_A"],
    },
    # ====== Theme 9: バッチ + CSTR ======
    {
        "case_id": "ms_batch_cstr_001",
        "context": "Determine the batch reactor's conversion dynamics and the equivalent CSTR's residence time for the same first-order reaction.",
        "equations": [
            "handbook_reaction_engineering__eq_1",   # N_A0 dX_A/dt = -r_A V
            "handbook_chemeng__eq_3",                # τ = (C_A0 - C_A) / (-r_A)
        ],
        # 2 方程式、2 未知数: t と τ
        "inputs": ["N_{A0}", "X_A", "-r_A", "V", "C_{A0}", "C_A", "r_A"],
        "outputs": ["t", "\\tau"],
    },
    # ====== Theme 10: PBR + 反応工学 ======
    {
        "case_id": "ms_pbr_chemeng_001",
        "context": "For a packed bed reactor, calculate the catalyst weight required (from the catalyst-based design equation) and the steady-state outlet concentration (from CSTR analogy).",
        "equations": [
            "handbook_reaction_engineering__eq_4",   # dX_A/dV = -r_A / F_A0
            "handbook_chemeng__eq_3",                # τ = (C_A0 - C_A) / (-r_A)
        ],
        "inputs": ["X_A", "V", "F_{A0}", "C_{A0}", "C_A", "r_A"],
        "outputs": ["-r_A", "\\tau"],
    },
    # ====== Theme 11: 反応 + 対流伝熱 ======
    {
        "case_id": "ms_reaction_convection_001",
        "context": "For a wall-cooled reactor, compute the convective heat removal from the reactor wall and the reactor residence time, given surface temperatures and concentration profile.",
        "equations": [
            "handbook_heat_transfer__eq_3",    # q_conv = h A_s (T_s - T_∞)
            "handbook_reaction_engineering__eq_3",  # τ = V/v_0 = C_A0 X_A / (-r_A)
        ],
        "inputs": ["h", "A_s", "T_s", "T_\\infty", "V", "v_0", "C_{A0}", "X_A", "-r_A"],
        "outputs": ["q_{conv}", "\\tau"],
    },
    # ====== Theme 12: PFR + 対流伝熱 ======
    {
        "case_id": "ms_pfr_convection_001",
        "context": "For a tubular reactor with wall cooling, compute the PFR volume and the convective heat removal, given the conversion profile, kinetics, and heat transfer characteristics.",
        "equations": [
            "handbook_reaction_engineering__eq_5",   # V = F_A0 ∫ dX_A / (-r_A)
            "handbook_heat_transfer__eq_3",          # q_conv = h A_s (T_s - T_∞)
        ],
        "inputs": ["F_{A0}", "X_{A,out}", "-r_A", "h", "A_s", "T_s", "T_\\infty"],
        "outputs": ["V", "q_{conv}"],
    },
    # ====== Theme 13: バイオ + 物質移動 ======
    {
        "case_id": "ms_bio_diffusion_001",
        "context": "In a bioreactor, calculate the Monod growth rate and the diffusion flux of nutrients, given the substrate concentration profile and bioreactor parameters.",
        "equations": [
            "handbook_bioprocess__eq_1",       # μ = μ_max S / (K_S + S)
            "handbook_mass_transfer__eq_1",    # J_Az = -D_AB dC_A/dz
        ],
        "inputs": ["\\mu_{max}", "S", "K_S", "D_{AB}", "C_A", "z"],
        "outputs": ["\\mu", "J_{Az}"],
    },
    # ====== Theme 14: PID + Arrhenius (温度制御) ======
    {
        "case_id": "ms_pid_arrhenius_001",
        "context": "For a temperature-controlled reactor, compute the Arrhenius rate constant at the current temperature and the PID controller output for maintaining the setpoint.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",  # Arrhenius
            "handbook_process_control__eq_1",   # PID
        ],
        "inputs": ["k_0", "E", "R", "T", "K_p", "K_i", "K_d", "e(t)", "r(t)", "y(t)", "t", "\\tau"],
        "outputs": ["k", "u(t)"],
    },
    # ====== Theme 15: PFR + Arrhenius + 反応速度 (3 ソース) ======
    {
        "case_id": "ms_pfr_arrhenius_rate_001",
        "context": "Size a PFR for a first-order Arrhenius reaction by combining: (1) Arrhenius rate constant, (2) first-order rate equation, (3) PFR integral design equation. This requires equations from three different sources.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",
            "Process_Control_Designing_Processes_and_Control_Systems_for_Dynamic_Performance.pdf__p0481-0560_eq_14.13_reaction_rate",
            "handbook_reaction_engineering__eq_5",   # V = F_A0 ∫ dX_A / (-r_A)
        ],
        # 3 方程式、3 未知数: k, r_a, V (-r_A は r_a と区別される変数なので、入力)
        "inputs": ["k_0", "E", "R", "T", "C_A", "F_{A0}", "X_{A,out}", "-r_A"],
        "outputs": ["k", "r_a", "V"],
    },
    # ====== Theme 16: バッチ + Arrhenius ======
    {
        "case_id": "ms_batch_arrhenius_001",
        "context": "For a temperature-dependent batch reaction, compute the Arrhenius rate constant and the batch reactor conversion dynamics.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",  # Arrhenius
            "handbook_reaction_engineering__eq_1",   # N_A0 dX_A/dt = -r_A V
        ],
        "inputs": ["k_0", "E", "R", "T", "N_{A0}", "X_A", "-r_A", "V"],
        "outputs": ["k", "t"],
    },
    # ====== Theme 17: バッチ反応器の積分形 ======
    {
        "case_id": "ms_batch_integral_001",
        "context": "Calculate the time required for a batch reactor (from the integral design equation) and the equivalent CSTR residence time, given identical kinetics and target conversion.",
        "equations": [
            "handbook_reaction_engineering__eq_2",   # t = N_A0 ∫ dX_A / (-r_A V)
            "handbook_chemeng__eq_3",                # τ = (C_A0 - C_A) / (-r_A)
        ],
        "inputs": ["N_{A0}", "X_A", "-r_A", "V", "C_{A0}", "C_A", "r_A"],
        "outputs": ["t", "\\tau"],
    },
    # ====== Theme 18: 全体伝熱 (ハンドブック chemeng) + 対流伝熱 (ハンドブック heat_transfer) ======
    {
        "case_id": "ms_chemeng_convection_001",
        "context": "For a CSTR with a cooling coil, compute the residence time from the reactor mass balance and the convective heat transfer rate at the coil surface.",
        "equations": [
            "handbook_chemeng__eq_3",          # τ = (C_A0 - C_A) / (-r_A)
            "handbook_heat_transfer__eq_3",    # q_conv = h A_s (T_s - T_∞)
        ],
        "inputs": ["C_{A0}", "C_A", "r_A", "h", "A_s", "T_s", "T_\\infty"],
        "outputs": ["\\tau", "q_{conv}"],
    },
    # ====== Theme 19: バッチ反応器 + FOPDT センサーモデル ======
    {
        "case_id": "ms_batch_sensor_001",
        "context": "For a batch reactor with a temperature sensor, compute the batch reaction time and the sensor's FOPDT transfer function, combining batch reactor engineering and process control.",
        "equations": [
            "handbook_reaction_engineering__eq_1",   # N_A0 dX_A/dt = -r_A V
            "handbook_process_control__eq_5",        # G_p(s) = K_p e^(-θs) / (τ_p s + 1)
        ],
        "inputs": ["N_{A0}", "X_A", "-r_A", "V", "K_p", "\\theta", "\\tau_p", "s"],
        "outputs": ["t", "G_p(s)"],
    },
    # ====== Theme 20: バイオプロセス + PID (細胞培養の制御) ======
    {
        "case_id": "ms_bio_pid_001",
        "context": "For a bioreactor with PID-controlled feed rate, compute the Monod-based specific growth rate and the controller output simultaneously.",
        "equations": [
            "handbook_bioprocess__eq_1",       # μ = μ_max S / (K_S + S)
            "handbook_process_control__eq_1",  # PID
        ],
        "inputs": ["\\mu_{max}", "S", "K_S", "K_p", "K_i", "K_d", "e(t)", "r(t)", "y(t)", "t", "\\tau"],
        "outputs": ["\\mu", "u(t)"],
    },
    # ====== Theme 21: PFR + 物質移動 ======
    {
        "case_id": "ms_pfr_masstransfer_001",
        "context": "For a packed bed reactor with external diffusion limitation, compute the PFR volume from the integral design equation and the mass transfer flux to the catalyst surface.",
        "equations": [
            "handbook_reaction_engineering__eq_5",  # V = F_A0 ∫ dX_A / (-r_A)
            "handbook_mass_transfer__eq_1",         # J_Az = -D_AB dC_A/dz
        ],
        "inputs": ["F_{A0}", "X_{A,out}", "-r_A", "D_{AB}", "C_A", "z"],
        "outputs": ["V", "J_{Az}"],
    },
    # ====== Theme 22: CSTR + 物質移動 ======
    {
        "case_id": "ms_cstr_masstransfer_001",
        "context": "For a CSTR with external mass transfer between liquid and gas phases, compute the residence time and the diffusion flux at the gas-liquid interface, given the outlet concentration.",
        "equations": [
            "handbook_chemeng__eq_3",           # τ = (C_A0 - C_A) / (-r_A)
            "handbook_mass_transfer__eq_1",     # J_Az = -D_AB dC_A/dz
        ],
        "inputs": ["C_{A0}", "C_A", "r_A", "D_{AB}", "z"],
        "outputs": ["\\tau", "J_{Az}"],
    },
    # ====== Theme 23: 反応 + 全体伝熱 + Arrhenius (3 ソース) ======
    {
        "case_id": "ms_reactor_heat_arrhenius_001",
        "context": "Three-source non-isothermal reactor design: combine the Arrhenius temperature dependence, the first-order rate expression, and the overall heat transfer law to determine the rate constant, reaction rate, and heat removal in a temperature-controlled CSTR.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",
            "Process_Control_Designing_Processes_and_Control_Systems_for_Dynamic_Performance.pdf__p0481-0560_eq_14.13_reaction_rate",
            "handbook_heat_transfer__eq_5",    # q = U A ΔT_overall
        ],
        "inputs": ["k_0", "E", "R", "T", "C_A", "U", "A", "\\Delta T_{overall}"],
        "outputs": ["k", "r_a", "q"],
    },
    # ====== Theme 24: PFR + バッチ + Arrhenius (3 ソース) ======
    {
        "case_id": "ms_pfr_batch_arrhenius_001",
        "context": "Design a temperature-controlled reactor by combining: (1) Arrhenius rate constant, (2) batch reactor design equation, (3) PFR integral design equation.",
        "equations": [
            "SEBORG_3rd_Edition_Process_Dynamics_and.pdf__p0081-0160_eq_2-63",
            "handbook_reaction_engineering__eq_1",   # batch
            "handbook_reaction_engineering__eq_5",   # PFR
        ],
        "inputs": ["k_0", "E", "R", "T", "N_{A0}", "X_A", "-r_A", "F_{A0}", "X_{A,out}"],
        "outputs": ["k", "t", "V"],
    },
    # ====== Theme 25: バッチ + 物質移動 ======
    {
        "case_id": "ms_batch_diffusion_001",
        "context": "For a batch reactor with controlled mass transfer from a feeding solid surface, compute the batch reaction time and the diffusion flux at the surface.",
        "equations": [
            "handbook_reaction_engineering__eq_1",   # N_A0 dX_A/dt = -r_A V
            "handbook_mass_transfer__eq_1",          # J_Az = -D_AB dC_A/dz
        ],
        "inputs": ["N_{A0}", "X_A", "-r_A", "V", "D_{AB}", "C_A", "z"],
        "outputs": ["t", "J_{Az}"],
    },
    # ====== Theme 26: バイオ + 物質移動 + Monod ======
    {
        "case_id": "ms_bio_diffusion_002",
        "context": "For an immobilized cell bioreactor with diffusion-limited substrate uptake, compute the Monod growth rate and the substrate diffusion flux into the cell, given the substrate concentration profile.",
        "equations": [
            "handbook_bioprocess__eq_1",       # μ = μ_max S / (K_S + S)
            "handbook_mass_transfer__eq_1",    # J_Az = -D_AB dC_A/dz
        ],
        "inputs": ["\\mu_{max}", "K_S", "D_{AB}", "z", "S", "C_A"],
        "outputs": ["\\mu", "J_{Az}"],
    },
    # ====== Theme 27: 反応 + 化学工学 (定義クロス) ======
    {
        "case_id": "ms_chemeng_re_001",
        "context": "Cross-validate the CSTR residence time using both the chemeng handbook (concentration-based) and reaction engineering handbook (conversion-based) formulations.",
        "equations": [
            "handbook_chemeng__eq_3",
            "handbook_reaction_engineering__eq_3",
        ],
        "inputs": ["C_{A0}", "r_A", "V", "v_0", "X_A", "-r_A"],
        "outputs": ["\\tau", "C_A"],
    },
    # ====== Theme 28: 全体伝熱 + 反応 (定常 CSTR モデル) ======
    {
        "case_id": "ms_steady_cstr_heat_001",
        "context": "For a steady-state isothermal CSTR with cooling: compute residence time and the corresponding heat removal needed, given concentration and temperature differentials.",
        "equations": [
            "handbook_reaction_engineering__eq_3",   # τ = V/v_0 = C_A0 X_A / (-r_A)
            "handbook_heat_transfer__eq_5",          # q = U A ΔT_overall
        ],
        "inputs": ["V", "v_0", "C_{A0}", "X_A", "-r_A", "U", "A", "\\Delta T_{overall}"],
        "outputs": ["\\tau", "q"],
    },
    # ====== Theme 29: PFR + PID 制御 ======
    {
        "case_id": "ms_pfr_pid_001",
        "context": "For a PFR with PID-controlled inlet flow rate, compute the required reactor volume and the PID controller output for setpoint tracking.",
        "equations": [
            "handbook_reaction_engineering__eq_5",   # V = F_A0 ∫ dX_A / (-r_A)
            "handbook_process_control__eq_1",        # PID
        ],
        "inputs": ["F_{A0}", "X_{A,out}", "-r_A", "K_p", "K_i", "K_d", "e(t)", "r(t)", "y(t)", "t", "\\tau"],
        "outputs": ["V", "u(t)"],
    },
    # ====== Theme 30: バッチ + 対流伝熱 (温度制御バッチ) ======
    {
        "case_id": "ms_batch_convection_001",
        "context": "For a jacketed batch reactor, compute the batch time required for a given conversion and the convective heat removal rate at the jacket.",
        "equations": [
            "handbook_reaction_engineering__eq_1",   # N_A0 dX_A/dt = -r_A V
            "handbook_heat_transfer__eq_3",          # q_conv = h A_s (T_s - T_∞)
        ],
        "inputs": ["N_{A0}", "X_A", "-r_A", "V", "h", "A_s", "T_s", "T_\\infty"],
        "outputs": ["t", "q_{conv}"],
    },
]


# ============================================================
# DoF=0 検証
# ============================================================

def load_equations() -> Dict[str, dict]:
    with open(EQUATIONS_JSON, encoding="utf-8") as f:
        eqs = json.load(f)
    return {f"{e['source_id']}__{e['eq_id']}": e for e in eqs}


def case_variables(eq_keys: List[str], eq_db: Dict[str, dict]) -> Set[str]:
    """正解式集合の変数の和集合"""
    vs: Set[str] = set()
    for k in eq_keys:
        e = eq_db.get(k)
        if e is None:
            raise ValueError(f"Equation {k} not found in DB")
        vs.update(e.get("variables", {}).keys())
    return vs


def check_solvability(case: dict, eq_db: Dict[str, dict]) -> tuple[bool, str]:
    """DoF=0 を含む 5 条件をチェック"""
    eq_keys = case["correct_model_ids"]
    inputs = set(case["input_variables"])
    outputs = set(case["output_variables"])

    missing = [k for k in eq_keys if k not in eq_db]
    if missing:
        return False, f"missing equations: {missing}"

    all_vars = case_variables(eq_keys, eq_db)

    bad_out = outputs - all_vars
    if bad_out:
        return False, f"outputs not in eq vars: {bad_out}"
    bad_in = inputs - all_vars
    if bad_in:
        return False, f"inputs not in eq vars: {bad_in}"

    unknowns = all_vars - inputs
    if not outputs <= unknowns:
        return False, f"output overlap with input: {outputs & inputs}"

    n_eq = len(eq_keys)
    if len(unknowns) != n_eq:
        return False, f"DoF != 0: |unknowns|={len(unknowns)}, n_eq={n_eq}, unknowns={sorted(unknowns)}, all_vars={sorted(all_vars)}"

    return True, "OK"


def n_sources(case: dict) -> int:
    return len({m.split("__", 1)[0] for m in case.get("correct_model_ids", [])})


def main():
    eq_db = load_equations()
    print(f"DB: {len(eq_db)} 式")
    print(f"テンプレート数: {len(TEMPLATES)}")
    print()

    cases = []
    fails = []
    for t in TEMPLATES:
        case = {
            "case_id": t["case_id"],
            "original_core_id": t["case_id"],
            "variant_type": "multisource_original",
            "context": t["context"],
            "input_variables": t["inputs"],
            "output_variables": t["outputs"],
            "correct_model_ids": t["equations"],
        }
        ok, msg = check_solvability(case, eq_db)
        if ok:
            cases.append(case)
            print(f"  OK {case['case_id']:35s} ({n_sources(case)} sources, {len(t['equations'])} eqs)")
        else:
            fails.append((case["case_id"], msg))
            print(f"  NG {case['case_id']:35s} -- {msg[:120]}")

    print()
    print(f"=== 結果 ===")
    print(f"成功: {len(cases)} / {len(TEMPLATES)} ケース")
    print(f"失敗: {len(fails)} ケース")

    from collections import Counter
    source_counts = [n_sources(c) for c in cases]
    print(f"\nソース横断度の分布:")
    for n, cnt in sorted(Counter(source_counts).items()):
        print(f"  {n} ソース: {cnt} ケース")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
