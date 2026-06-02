"""
データセット拡大スクリプト: 1,613式 → 3,000+式を目指す。

モード:
  --mode handbook    : LLM知識から基礎式を生成 (Round 1: 15ドメイン)
  --mode handbook2   : ハンドブック第2弾 (新10ドメイン + 既存16ドメインの応用式)
  --mode papers      : Semantic Scholar で論文探索 + open-access PDF ダウンロード
  --mode process     : 未処理PDFの数式抽出
  --mode split-large : 大型PDF(>1000p)をチャンク分割して抽出
  --mode all         : 上記すべてを順に実行

使い方:
  python expand_dataset.py --mode handbook2         # ハンドブック第2弾
  python expand_dataset.py --mode split-large       # 大型PDF分割抽出
  python expand_dataset.py --mode all               # 全部
  python expand_dataset.py --mode handbook2 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

import requests

# ---------------------------------------------------------------------------
# パス定数
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
PROCESSED_PDFS_FILE = ROOT / "processed_pdfs.json"
INPUT_PDF_DIR = ROOT / "input_pdfs"
PAPER_LINKS_FILE = ROOT / "paper_dataset_links.json"

# ---------------------------------------------------------------------------
# ハンドブック拡張: ドメイン定義
# ---------------------------------------------------------------------------

HANDBOOK_DOMAINS: List[Dict[str, str]] = [
    {
        "source_id": "handbook_fluid_dynamics",
        "domain_label": "Fluid Dynamics",
        "topics": (
            "Navier-Stokes equations (incompressible, compressible), "
            "Bernoulli equation, Hagen-Poiseuille flow, Darcy-Weisbach friction, "
            "boundary layer equations (Blasius), drag and lift coefficients, "
            "continuity equation, Euler equation, potential flow, vorticity equation, "
            "turbulent flow (Reynolds decomposition, k-epsilon model basics), "
            "open channel flow (Manning, Chezy), orifice flow, Torricelli's theorem"
        ),
        "n_equations": 25,
    },
    {
        "source_id": "handbook_heat_transfer",
        "domain_label": "Heat Transfer",
        "topics": (
            "Fourier's law (1D, 3D), Newton's law of cooling, "
            "Stefan-Boltzmann radiation law, overall heat transfer coefficient, "
            "fin efficiency and fin heat transfer, lumped capacitance model, "
            "thermal resistance analogy, LMTD and epsilon-NTU methods for heat exchangers, "
            "Dittus-Boelter correlation, Sieder-Tate correlation, "
            "Nusselt number definitions, Biot number criterion, "
            "thermal boundary layer, boiling and condensation correlations"
        ),
        "n_equations": 25,
    },
    {
        "source_id": "handbook_mass_transfer",
        "domain_label": "Mass Transfer",
        "topics": (
            "Fick's first and second law of diffusion, "
            "convective mass transfer coefficient, film theory, "
            "penetration theory (Higbie), surface renewal theory, "
            "mass transfer with chemical reaction, "
            "absorption tower design (HTU, NTU), "
            "diffusion in porous media (effective diffusivity), "
            "Sherwood number correlations, Schmidt number, "
            "membrane transport (solution-diffusion model), "
            "binary distillation (McCabe-Thiele basics)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_thermodynamics",
        "domain_label": "Thermodynamics",
        "topics": (
            "ideal gas law, van der Waals equation, "
            "Clausius-Clapeyron equation, Antoine equation, "
            "Raoult's law, Henry's law, "
            "first law (closed/open system), second law (entropy balance), "
            "Gibbs free energy, chemical potential, "
            "fugacity and activity coefficient, "
            "Carnot efficiency, Rankine cycle, "
            "flash calculations (Rachford-Rice), "
            "mixing rules for multi-component systems"
        ),
        "n_equations": 25,
    },
    {
        "source_id": "handbook_reaction_engineering",
        "domain_label": "Reaction Engineering",
        "topics": (
            "batch reactor design equation, "
            "PFR (plug flow reactor) design equation, "
            "CSTR design equation with multiple reactions, "
            "Arrhenius equation and modified Arrhenius, "
            "selectivity and yield for parallel/series reactions, "
            "Thiele modulus and effectiveness factor, "
            "Langmuir-Hinshelwood kinetics, Michaelis-Menten kinetics, "
            "adiabatic reactor temperature rise, "
            "residence time distribution (E(t), F(t)), "
            "catalytic fixed-bed reactor (Ergun equation for pressure drop)"
        ),
        "n_equations": 25,
    },
    {
        "source_id": "handbook_process_control",
        "domain_label": "Process Control",
        "topics": (
            "PID controller equation (positional, velocity forms), "
            "first-order plus dead-time (FOPDT) transfer function, "
            "second-order transfer function (underdamped, critically damped), "
            "Ziegler-Nichols tuning rules, Cohen-Coon tuning, "
            "Routh-Hurwitz stability criterion, "
            "Bode stability margins (gain margin, phase margin), "
            "Nyquist criterion, root locus gain equation, "
            "Smith predictor for time delay compensation, "
            "IMC (Internal Model Control) design, "
            "cascade control, feedforward control"
        ),
        "n_equations": 25,
    },
    {
        "source_id": "handbook_electrochemistry",
        "domain_label": "Electrochemistry",
        "topics": (
            "Nernst equation, Butler-Volmer equation, "
            "Tafel equation (anodic and cathodic), "
            "Faraday's law of electrolysis, "
            "ohmic drop in electrolyte, "
            "Warburg impedance, Randles equivalent circuit, "
            "concentration overpotential, "
            "lithium-ion battery voltage models (OCV + internal resistance), "
            "fuel cell polarization curve, "
            "corrosion rate from Tafel slopes, "
            "electrochemical impedance spectroscopy basics"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_bioprocess",
        "domain_label": "Bioprocess Engineering",
        "topics": (
            "Monod kinetics (specific growth rate), "
            "Luedeking-Piret model for product formation, "
            "chemostat (CSTR with biomass) steady-state equations, "
            "batch fermentation growth phases, "
            "oxygen transfer rate (OTR = kLa * (C* - C)), "
            "substrate inhibition (Andrews/Haldane model), "
            "cell death and maintenance energy, "
            "logistic growth equation, "
            "yield coefficients (Y_X/S, Y_P/S), "
            "fed-batch mass balance"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_environmental",
        "domain_label": "Environmental Engineering",
        "topics": (
            "BOD kinetics (first-order decay), "
            "Streeter-Phelps oxygen sag model, "
            "activated sludge model (mass balance for biomass/substrate), "
            "sedimentation (Stokes law for terminal velocity), "
            "filtration equation (Carmen-Kozeny), "
            "adsorption isotherms (Langmuir, Freundlich, BET), "
            "chlorine decay kinetics, "
            "Gaussian plume model for air pollution dispersion, "
            "groundwater flow (Darcy's law), "
            "landfill gas generation model"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_vibrations",
        "domain_label": "Mechanical Vibrations",
        "topics": (
            "undamped free vibration (spring-mass), "
            "damped free vibration (underdamped, overdamped, critically damped), "
            "forced vibration with harmonic excitation, "
            "transmissibility ratio, vibration isolation, "
            "natural frequency of beams (Euler-Bernoulli), "
            "torsional vibration, "
            "two-degree-of-freedom system (coupled oscillators), "
            "modal analysis basics, "
            "Rayleigh quotient for natural frequency estimation, "
            "rotating unbalance"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_electrical",
        "domain_label": "Electrical Engineering",
        "topics": (
            "RLC circuit differential equations (series, parallel), "
            "AC circuit impedance and power (real, reactive, apparent), "
            "transformer equations (turns ratio, impedance reflection), "
            "DC motor equations (back-EMF, torque-speed), "
            "three-phase power, "
            "RC and RL time constants, "
            "resonant frequency, quality factor Q, "
            "Kirchhoff's voltage and current laws, "
            "op-amp transfer functions (inverting, non-inverting, integrator), "
            "power electronics (buck converter steady-state)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_polymer",
        "domain_label": "Polymer Engineering",
        "topics": (
            "free radical polymerization kinetics (initiation, propagation, termination), "
            "degree of polymerization (number-average, weight-average), "
            "Flory-Huggins equation (mixing free energy), "
            "Mark-Houwink equation (intrinsic viscosity), "
            "power-law viscosity model, Cross model, "
            "rubber elasticity (neo-Hookean), "
            "Avrami equation for crystallization kinetics, "
            "WLF equation for temperature-viscosity shift, "
            "step-growth polymerization (Carothers equation), "
            "copolymerization equation (Mayo-Lewis)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_combustion",
        "domain_label": "Combustion Engineering",
        "topics": (
            "stoichiometric air-fuel ratio, "
            "adiabatic flame temperature, "
            "laminar flame speed (Metghalchi-Keck correlation), "
            "Damkohler number for flame stability, "
            "NOx formation (Zeldovich mechanism), "
            "droplet evaporation (d-squared law), "
            "detonation Chapman-Jouguet relations, "
            "premixed flame (Rankine-Hugoniot), "
            "radiation from gas flames (optically thin model), "
            "explosion limits (chain branching kinetics)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_separation",
        "domain_label": "Separation Processes",
        "topics": (
            "McCabe-Thiele (operating line, q-line, minimum reflux), "
            "Fenske equation (minimum stages), "
            "Underwood equation (minimum reflux), "
            "extraction (distribution coefficient, number of stages), "
            "membrane separation (flux, rejection, concentration polarization), "
            "crystallization kinetics (nucleation rate, crystal growth rate), "
            "drying curves (constant rate, falling rate), "
            "evaporator design (single and multiple effect), "
            "leaching (rate equation, Shanks system), "
            "centrifuge separation (sigma theory)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_transport_phenomena",
        "domain_label": "Transport Phenomena",
        "topics": (
            "general transport equation (Bird-Stewart-Lightfoot), "
            "momentum transport (Newton's law of viscosity, non-Newtonian models), "
            "energy transport (Fourier), mass transport (Fick), "
            "shell balance methodology for pipe flow, "
            "dimensionless groups (Re, Pr, Sc, Pe, Le, Da, Bi, Nu, Sh), "
            "analogy between heat and mass transfer (Chilton-Colburn), "
            "boundary conditions for transport equations, "
            "turbulent transport (eddy diffusivity), "
            "multicomponent diffusion (Stefan-Maxwell), "
            "coupled heat and mass transfer (wet bulb temperature)"
        ),
        "n_equations": 25,
    },
]

# ---------------------------------------------------------------------------
# ハンドブック拡張 Round 2: 新ドメイン + 既存ドメインの応用式
# ---------------------------------------------------------------------------

HANDBOOK_DOMAINS_R2: List[Dict[str, str]] = [
    # --- 新ドメイン 10 種 ---
    {
        "source_id": "handbook_nuclear",
        "domain_label": "Nuclear Engineering",
        "topics": (
            "neutron diffusion equation (one-group, two-group), "
            "criticality condition (k-effective), "
            "point kinetics equations (delayed neutrons), "
            "radioactive decay law, decay chains (Bateman equations), "
            "heat generation in fuel rods, "
            "reactor thermal-hydraulics (Dittus-Boelter for coolant), "
            "neutron moderation (slowing-down), "
            "burnup equations, "
            "dose rate calculations (inverse square law, shielding), "
            "xenon poisoning transient"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_semiconductor",
        "domain_label": "Semiconductor Physics",
        "topics": (
            "drift-diffusion equations for carriers, "
            "Poisson equation for electrostatics, "
            "Shockley diode equation, "
            "MOSFET drain current (linear, saturation), "
            "threshold voltage, "
            "carrier recombination (SRH, Auger, radiative), "
            "diffusion of dopants (Fick's law applied), "
            "Deal-Grove oxidation model, "
            "Einstein relation for mobility-diffusivity, "
            "depletion width and capacitance of p-n junction"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_acoustics",
        "domain_label": "Acoustics",
        "topics": (
            "wave equation (1D, 3D), "
            "plane wave propagation (pressure, velocity), "
            "sound intensity and power, "
            "decibel scale (SPL, SWL, SIL), "
            "Helmholtz resonator frequency, "
            "room acoustics (Sabine reverberation time), "
            "transmission loss through walls, "
            "Doppler effect, "
            "acoustic impedance, reflection and transmission coefficients, "
            "standing waves in pipes (open, closed)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_optics",
        "domain_label": "Optics and Photonics",
        "topics": (
            "Snell's law, Fresnel equations, "
            "thin lens equation, lensmaker's equation, "
            "diffraction grating equation, "
            "Rayleigh criterion for resolution, "
            "Beer-Lambert law for absorption, "
            "Planck's radiation law, Wien's displacement law, "
            "laser rate equations (two-level, three-level), "
            "Gaussian beam propagation (waist, divergence, Rayleigh range), "
            "fiber optics (numerical aperture, attenuation), "
            "Fabry-Perot interferometer (free spectral range, finesse)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_geotechnical",
        "domain_label": "Geotechnical Engineering",
        "topics": (
            "Terzaghi's consolidation theory, "
            "Mohr-Coulomb failure criterion, "
            "Darcy's law for seepage, "
            "effective stress principle, "
            "bearing capacity equations (Terzaghi, Meyerhof), "
            "earth pressure (Rankine, Coulomb), "
            "settlement calculations (elastic, consolidation), "
            "coefficient of permeability, "
            "slope stability (factor of safety), "
            "compaction curve (Proctor test relationship)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_aerodynamics",
        "domain_label": "Aerodynamics",
        "topics": (
            "lift and drag coefficients, "
            "thin airfoil theory (lift slope), "
            "induced drag (finite wing, Oswald efficiency), "
            "Prandtl-Glauert compressibility correction, "
            "shock relations (normal shock, oblique shock), "
            "isentropic flow relations, "
            "Mach number and speed of sound, "
            "Prandtl-Meyer expansion, "
            "boundary layer (laminar vs turbulent transition, skin friction), "
            "propeller thrust and efficiency"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_hvac",
        "domain_label": "HVAC and Refrigeration",
        "topics": (
            "psychrometric relations (humidity ratio, enthalpy of moist air), "
            "cooling/heating load calculation, "
            "vapor-compression refrigeration cycle (COP), "
            "evaporator and condenser heat transfer, "
            "duct sizing (friction loss, Darcy-Weisbach for air), "
            "fan laws (affinity laws), "
            "thermal comfort (PMV-PPD model basics), "
            "heat pump performance, "
            "air change rate and ventilation, "
            "absorption refrigeration (COP, generator-absorber balance)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_food_process",
        "domain_label": "Food Process Engineering",
        "topics": (
            "thermal death time (D-value, z-value, F-value), "
            "Arrhenius for microbial inactivation, "
            "water activity and sorption isotherms (GAB model), "
            "drying kinetics (thin-layer models, Page equation), "
            "freezing time estimation (Plank's equation), "
            "rheology of food (power-law, Herschel-Bulkley), "
            "heat penetration in cans (Ball's formula), "
            "enzyme inactivation kinetics, "
            "mass transfer in osmotic dehydration, "
            "quality degradation kinetics (vitamin loss, color change)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_pharma",
        "domain_label": "Pharmaceutical Engineering",
        "topics": (
            "pharmacokinetics (one-compartment, two-compartment models), "
            "drug dissolution (Noyes-Whitney equation), "
            "Michaelis-Menten enzyme kinetics, "
            "Higuchi model for drug release, "
            "first-order absorption and elimination, "
            "bioavailability and area under curve (AUC), "
            "clearance and volume of distribution, "
            "dose-response (Hill equation, Emax model), "
            "diffusion through membranes (Fick), "
            "crystallization in pharmaceutical systems"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_structural",
        "domain_label": "Structural Mechanics",
        "topics": (
            "Euler buckling load, "
            "beam bending (Euler-Bernoulli, moment-curvature), "
            "shear stress in beams, "
            "Mohr's circle for stress transformation, "
            "deflection formulas (cantilever, simply supported), "
            "torsion of circular shafts, "
            "combined loading (von Mises yield criterion), "
            "fatigue life (S-N curve, Basquin's law), "
            "stress concentration factor, "
            "plate bending (Kirchhoff theory basics)"
        ),
        "n_equations": 20,
    },
    # --- 既存ドメインの応用・上級式 ---
    {
        "source_id": "handbook_fluid_dynamics_adv",
        "domain_label": "Fluid Dynamics: Advanced",
        "topics": (
            "compressible flow (isentropic relations, choked flow), "
            "shock tube equations, "
            "lubrication theory (Reynolds equation), "
            "Stokes flow (creeping motion around sphere), "
            "cavitation number and inception, "
            "two-phase flow (drift flux model, Lockhart-Martinelli), "
            "non-Newtonian flow (power-law in pipe, Bingham plastic), "
            "flow in packed beds (Ergun extended), "
            "vortex shedding (Strouhal number), "
            "water hammer (Joukowsky equation)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_heat_transfer_adv",
        "domain_label": "Heat Transfer: Advanced",
        "topics": (
            "radiative exchange (view factors, radiosity method), "
            "combined radiation and convection, "
            "transient conduction (semi-infinite solid, Heisler charts), "
            "heat transfer in packed beds, "
            "phase change (Stefan problem, enthalpy method), "
            "micro-scale heat transfer (Knudsen number effects), "
            "heat pipes (capillary limit, wick performance), "
            "extended surfaces (annular fins, spine fins), "
            "jet impingement heat transfer correlations, "
            "thermal contact resistance"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_reaction_eng_adv",
        "domain_label": "Reaction Engineering: Advanced",
        "topics": (
            "non-isothermal reactor design (energy balance coupled with mole balance), "
            "multiple steady states in CSTR (ignition/extinction), "
            "fluidized bed reactor (two-phase model), "
            "membrane reactor (permeation + reaction), "
            "photocatalytic reactor (LVRPA, radiation field), "
            "gas-liquid reactions (Hatta number, enhancement factor), "
            "deactivation kinetics (sintering, coking, poisoning), "
            "RTD for non-ideal reactors (tanks-in-series, dispersion model), "
            "microreactor (mass transfer limited regime), "
            "enzymatic reactor (immobilized enzyme, effectiveness)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_thermodynamics_adv",
        "domain_label": "Thermodynamics: Advanced",
        "topics": (
            "equations of state (Peng-Robinson, Soave-Redlich-Kwong), "
            "fugacity coefficients from cubic EOS, "
            "activity coefficient models (Margules, van Laar, Wilson, NRTL, UNIQUAC), "
            "VLE calculations (bubble point, dew point), "
            "chemical equilibrium (equilibrium constant, Gibbs minimization), "
            "excess properties and mixing rules, "
            "Joule-Thomson coefficient, "
            "real gas thermodynamic properties (departure functions), "
            "Clapeyron equation for solid-liquid-vapor transitions, "
            "multicomponent flash (Rachford-Rice extended)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_process_control_adv",
        "domain_label": "Process Control: Advanced",
        "topics": (
            "model predictive control (MPC, discrete-time state-space), "
            "state estimation (Kalman filter equations), "
            "adaptive control (MIT rule, recursive least squares), "
            "multivariable control (RGA, relative gain array), "
            "decoupling control, "
            "discrete-time systems (z-transform, sampling), "
            "robust control basics (H-infinity, structured singular value), "
            "nonlinear control (feedback linearization), "
            "observer design (Luenberger observer), "
            "inferential control (soft sensor)"
        ),
        "n_equations": 20,
    },
    {
        "source_id": "handbook_electrochemistry_adv",
        "domain_label": "Electrochemistry: Advanced",
        "topics": (
            "porous electrode theory (Newman), "
            "lithium-ion battery (P2D model: solid diffusion, electrolyte transport), "
            "solid electrolyte interphase (SEI) growth kinetics, "
            "supercapacitor (double-layer and pseudocapacitance models), "
            "electrolysis cell (overpotential components breakdown), "
            "rotating disk electrode (Levich equation), "
            "AC impedance (Nyquist analysis, Bode), "
            "corrosion (mixed potential theory, Evans diagram), "
            "flow battery (crossover, self-discharge), "
            "photoelectrochemistry (Gartner equation)"
        ),
        "n_equations": 20,
    },
]

# Semantic Scholar 検索クエリ（新ドメイン向け）
PAPER_QUERIES: List[Dict] = [
    {"query": "fluid dynamics pipe flow mathematical model equations", "limit": 30},
    {"query": "heat exchanger dynamic model differential equations", "limit": 30},
    {"query": "electrochemical cell battery model equations", "limit": 30},
    {"query": "bioprocess fermentation kinetic model mathematical", "limit": 30},
    {"query": "wastewater treatment activated sludge mathematical model", "limit": 30},
    {"query": "polymer reactor kinetics mathematical model", "limit": 30},
    {"query": "combustion flame speed model mathematical equations", "limit": 30},
    {"query": "distillation column dynamic simulation mathematical", "limit": 30},
    {"query": "membrane separation transport model mathematical", "limit": 30},
    {"query": "PID controller tuning process model mathematical", "limit": 30},
    {"query": "mass transfer absorption mathematical model equations", "limit": 30},
    {"query": "crystallization population balance mathematical model", "limit": 30},
    {"query": "vibration mechanical system dynamic equations model", "limit": 20},
    {"query": "chemical vapor deposition reactor model equations", "limit": 20},
    {"query": "power electronics converter mathematical model equations", "limit": 20},
]


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def load_unified(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_unified(data: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(data)} equations to {path}")


def existing_keys(data: list[dict]) -> set[tuple[str, str]]:
    return {(d["source_id"], d["eq_id"]) for d in data}


# ---------------------------------------------------------------------------
# Mode: handbook — LLM知識から基礎式を大量生成
# ---------------------------------------------------------------------------

def _build_handbook_prompt(domain: Dict) -> str:
    n = domain["n_equations"]
    source_id = domain["source_id"]
    label = domain["domain_label"]
    topics = domain["topics"]

    return f"""You are an expert in {label} and mathematical modeling.
Generate exactly {n} fundamental equations used in {label}, in LaTeX format.

Cover these topics:
{topics}

For each equation provide:
- source_id: use exactly "{source_id}"
- eq_id: "eq_1" through "eq_{n}" (sequential)
- equation: LaTeX form of the equation (use standard notation)
- variables: object with each variable symbol as key and its definition (with units where appropriate) as value
- context_text: brief physical meaning or context (1-3 sentences, include when and where this equation is used)
- domain: "{label}" or a more specific sub-domain (e.g., "{label}: Pipe Flow")

IMPORTANT:
- Each equation MUST be distinct (no duplicates).
- Include both the differential form and algebraic form where applicable.
- Use standard notation conventions for the field.
- Variables dict must include ALL symbols that appear in the equation.

Output a single JSON array of exactly {n} equation objects. No markdown fences."""


def generate_handbook_domain(domain: Dict, dry_run: bool = False) -> list[dict]:
    """1つのドメインについてLLMから基礎式を生成する。"""
    source_id = domain["source_id"]
    n = domain["n_equations"]
    label = domain["domain_label"]

    if dry_run:
        print(f"  [DRY RUN] Would generate {n} equations for {label} ({source_id})")
        return []

    from google import genai
    from extract_equations import _get_response_schema

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
    client = genai.Client(api_key=api_key)

    prompt = _build_handbook_prompt(domain)

    # 5 RPM 対策
    time.sleep(13)

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": _get_response_schema(),
        },
    )

    if hasattr(response, "text") and response.text:
        text = response.text.strip()
    elif response.candidates and response.candidates[0].content.parts:
        text = response.candidates[0].content.parts[0].text.strip()
    else:
        print(f"  WARNING: Empty response for {label}")
        return []

    # コードフェンス除去
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    raw_list = json.loads(text)
    if not isinstance(raw_list, list):
        raw_list = [raw_list]

    # source_id の強制上書き
    equations = []
    for item in raw_list:
        item["source_id"] = source_id
        equations.append(item)

    print(f"  Generated {len(equations)} equations for {label}")
    return equations


def run_handbook_expansion(dry_run: bool = False) -> int:
    """全ドメインについてハンドブック式を生成し、unified_equations.json に追記。"""
    print("=" * 60)
    print("HANDBOOK EXPANSION")
    print("=" * 60)

    data = load_unified(UNIFIED_JSON)
    keys = existing_keys(data)
    total_before = len(data)

    target_total = sum(d["n_equations"] for d in HANDBOOK_DOMAINS)
    print(f"Target: {target_total} new equations across {len(HANDBOOK_DOMAINS)} domains")
    print(f"Current total: {total_before} equations\n")

    # 既に生成済みのドメインをスキップ
    existing_sources = {d["source_id"] for d in data}

    added_total = 0
    for i, domain in enumerate(HANDBOOK_DOMAINS):
        source_id = domain["source_id"]
        label = domain["domain_label"]

        if source_id in existing_sources:
            count = sum(1 for d in data if d["source_id"] == source_id)
            print(f"[{i+1}/{len(HANDBOOK_DOMAINS)}] {label}: SKIP (already {count} equations)")
            continue

        print(f"[{i+1}/{len(HANDBOOK_DOMAINS)}] {label}...")
        try:
            new_eqs = generate_handbook_domain(domain, dry_run=dry_run)
            for eq in new_eqs:
                k = (eq["source_id"], eq["eq_id"])
                if k not in keys:
                    data.append(eq)
                    keys.add(k)
                    added_total += 1

            if not dry_run and new_eqs:
                save_unified(data, UNIFIED_JSON)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nHandbook expansion done: {total_before} → {len(data)} equations (+{added_total})")
    return added_total


def run_handbook_expansion_r2(dry_run: bool = False) -> int:
    """Round 2: 新ドメイン + 既存ドメインの応用式。"""
    print("=" * 60)
    print("HANDBOOK EXPANSION Round 2")
    print("=" * 60)

    data = load_unified(UNIFIED_JSON)
    keys = existing_keys(data)
    total_before = len(data)

    target_total = sum(d["n_equations"] for d in HANDBOOK_DOMAINS_R2)
    print(f"Target: {target_total} new equations across {len(HANDBOOK_DOMAINS_R2)} domains")
    print(f"Current total: {total_before} equations\n")

    existing_sources = {d["source_id"] for d in data}

    added_total = 0
    for i, domain in enumerate(HANDBOOK_DOMAINS_R2):
        source_id = domain["source_id"]
        label = domain["domain_label"]

        if source_id in existing_sources:
            count = sum(1 for d in data if d["source_id"] == source_id)
            print(f"[{i+1}/{len(HANDBOOK_DOMAINS_R2)}] {label}: SKIP (already {count} equations)")
            continue

        print(f"[{i+1}/{len(HANDBOOK_DOMAINS_R2)}] {label}...")
        try:
            new_eqs = generate_handbook_domain(domain, dry_run=dry_run)
            for eq in new_eqs:
                k = (eq["source_id"], eq["eq_id"])
                if k not in keys:
                    data.append(eq)
                    keys.add(k)
                    added_total += 1

            if not dry_run and new_eqs:
                save_unified(data, UNIFIED_JSON)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\nHandbook R2 done: {total_before} → {len(data)} equations (+{added_total})")
    return added_total


# ---------------------------------------------------------------------------
# Mode: split-large — 大型PDFをチャンク分割して抽出
# ---------------------------------------------------------------------------

LARGE_PDF_CHUNK_PAGES = 80  # 80ページずつ分割
LARGE_PDF_TARGETS = [
    "ChemicalProcessDynamicsAndControls.pdf",
    "SEBORG_3rd_Edition_Process_Dynamics_and.pdf",
    "Process_Control_Designing_Processes_and_Control_Systems_for_Dynamic_Performance.pdf",
]


def split_pdf_to_chunks(pdf_path: Path, chunk_size: int = LARGE_PDF_CHUNK_PAGES) -> list[Path]:
    """大型PDFを chunk_size ページごとに分割し、一時ファイルとして保存。"""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    print(f"  Total pages: {total_pages}")

    chunks_dir = INPUT_PDF_DIR / "_chunks"
    chunks_dir.mkdir(exist_ok=True)

    stem = pdf_path.stem[:40]  # ファイル名を短縮
    chunk_paths = []

    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size, total_pages)
        chunk_name = f"{stem}_p{start+1:04d}-{end:04d}.pdf"
        chunk_path = chunks_dir / chunk_name

        if chunk_path.exists():
            chunk_paths.append(chunk_path)
            continue

        writer = PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])

        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunk_paths.append(chunk_path)

    print(f"  Split into {len(chunk_paths)} chunks of ~{chunk_size} pages")
    return chunk_paths


def run_split_large_pdfs(dry_run: bool = False) -> int:
    """大型PDFをチャンク分割してから抽出。"""
    print("=" * 60)
    print("SPLIT & PROCESS LARGE PDFs")
    print("=" * 60)

    from extract_equations import extract_equations

    data = load_unified(UNIFIED_JSON)
    keys = existing_keys(data)
    total_before = len(data)

    # 処理済みチャンクの管理
    chunks_done_file = ROOT / "processed_chunks.json"
    if chunks_done_file.is_file():
        with open(chunks_done_file) as f:
            chunks_done = set(json.load(f))
    else:
        chunks_done = set()

    added_total = 0

    for pdf_name in LARGE_PDF_TARGETS:
        pdf_path = INPUT_PDF_DIR / pdf_name
        if not pdf_path.exists():
            print(f"\n{pdf_name}: NOT FOUND, skipping")
            continue

        print(f"\n{'─'*50}")
        print(f"Processing: {pdf_name}")

        if dry_run:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            n_pages = len(reader.pages)
            n_chunks = (n_pages + LARGE_PDF_CHUNK_PAGES - 1) // LARGE_PDF_CHUNK_PAGES
            print(f"  [DRY RUN] {n_pages} pages → {n_chunks} chunks")
            continue

        try:
            chunk_paths = split_pdf_to_chunks(pdf_path)
        except Exception as e:
            print(f"  Split ERROR: {e}")
            continue

        for j, chunk_path in enumerate(chunk_paths):
            if chunk_path.name in chunks_done:
                continue

            print(f"  [{j+1}/{len(chunk_paths)}] {chunk_path.name}")
            try:
                equations = extract_equations(chunk_path)
                source_id = pdf_name  # 元PDFのファイル名を source_id に
                added = 0
                for eq_dict in [e.model_dump() for e in equations]:
                    # eq_id にチャンク識別子を付与して重複回避
                    eq_dict["source_id"] = source_id
                    orig_eq_id = eq_dict["eq_id"]
                    chunk_label = chunk_path.stem.split("_p")[-1]  # e.g. "0001-0080"
                    eq_dict["eq_id"] = f"p{chunk_label}_{orig_eq_id}"

                    k = (eq_dict["source_id"], eq_dict["eq_id"])
                    if k not in keys:
                        data.append(eq_dict)
                        keys.add(k)
                        added += 1

                added_total += added
                print(f"    Extracted {len(equations)}, {added} new")

                save_unified(data, UNIFIED_JSON)
                chunks_done.add(chunk_path.name)
                with open(chunks_done_file, "w") as f:
                    json.dump(sorted(chunks_done), f)

            except Exception as e:
                print(f"    ERROR: {e}")
                # 503 は後でリトライ可能なのでスキップして次へ
                chunks_done.add(chunk_path.name)
                with open(chunks_done_file, "w") as f:
                    json.dump(sorted(chunks_done), f)

            time.sleep(15)  # 5 RPM

    print(f"\nLarge PDF processing done: {total_before} → {len(data)} equations (+{added_total})")
    return added_total


# ---------------------------------------------------------------------------
# Mode: papers — Semantic Scholar 検索 + PDF ダウンロード
# ---------------------------------------------------------------------------

SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,openAccessPdf,url"


def fetch_papers_for_query(query: str, limit: int, api_key: str | None = None) -> list[dict]:
    """1クエリで論文を取得。"""
    params = {"query": query, "limit": limit, "offset": 0, "fields": FIELDS}
    headers = {"x-api-key": api_key} if api_key else {}

    for attempt in range(4):
        try:
            resp = requests.get(SEARCH_URL, params=params, headers=headers or None, timeout=30)
        except requests.RequestException as e:
            print(f"    Network error: {e}")
            return []

        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"    429 rate limit. Waiting {wait}s...")
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            print(f"    HTTP {resp.status_code}: {resp.text[:200]}")
            return []
        result = resp.json()
        return result.get("data", []) if isinstance(result, dict) else result

    return []


def download_pdf(url: str, save_path: Path) -> bool:
    """PDFをダウンロード。"""
    try:
        resp = requests.get(url, timeout=60, stream=True)
        if resp.status_code != 200:
            return False
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type and "octet" not in content_type:
            # PDFでない可能性（HTMLリダイレクト等）
            first_bytes = resp.content[:10]
            if not first_bytes.startswith(b"%PDF"):
                return False
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    Download error: {e}")
        return False


def run_paper_search(dry_run: bool = False) -> int:
    """Semantic Scholar で論文を検索し、open-access PDFをダウンロード。"""
    print("=" * 60)
    print("PAPER SEARCH & DOWNLOAD")
    print("=" * 60)

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY")
    if api_key:
        print("Using Semantic Scholar API key")

    # 既にダウンロード済みのPDF名
    existing_pdfs = set()
    if INPUT_PDF_DIR.is_dir():
        existing_pdfs = {p.name for p in INPUT_PDF_DIR.glob("*.pdf")}
    print(f"Existing PDFs: {len(existing_pdfs)}")

    all_papers: list[dict] = []
    seen_ids: set[str] = set()
    downloaded = 0

    for i, qinfo in enumerate(PAPER_QUERIES):
        query = qinfo["query"]
        limit = qinfo["limit"]
        print(f"\n[{i+1}/{len(PAPER_QUERIES)}] Query: {query[:60]}...")

        papers = fetch_papers_for_query(query, limit, api_key)
        print(f"  Found {len(papers)} papers")

        for p in papers:
            pid = p.get("paperId")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)

            oa = p.get("openAccessPdf")
            pdf_url = None
            if isinstance(oa, dict) and oa.get("url"):
                pdf_url = oa["url"]

            paper_info = {
                "paperId": pid,
                "title": p.get("title", ""),
                "year": p.get("year"),
                "query": query[:40],
                "pdf_url": pdf_url,
            }
            all_papers.append(paper_info)

            # PDFダウンロード
            if pdf_url and not dry_run:
                # ファイル名を生成（安全な名前）
                safe_title = "".join(
                    c if c.isalnum() or c in "._- " else "_"
                    for c in (p.get("title") or pid)[:60]
                ).strip()
                year = p.get("year", "")
                fname = f"{year}_{safe_title}.pdf".replace(" ", "_")

                if fname in existing_pdfs:
                    continue

                save_path = INPUT_PDF_DIR / fname
                print(f"  Downloading: {fname[:50]}...")
                if download_pdf(pdf_url, save_path):
                    downloaded += 1
                    existing_pdfs.add(fname)
                    print(f"    OK ({save_path.stat().st_size / 1024:.0f} KB)")
                else:
                    print(f"    FAILED")
                time.sleep(1)  # 礼儀正しくダウンロード

        # クエリ間に少し待つ
        if i < len(PAPER_QUERIES) - 1:
            time.sleep(2)

    # 結果を保存
    with_pdf = sum(1 for p in all_papers if p.get("pdf_url"))
    result = {
        "total_papers_found": len(all_papers),
        "papers_with_pdf": with_pdf,
        "papers_downloaded": downloaded,
        "papers": all_papers,
    }
    with open(PAPER_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nPaper search done:")
    print(f"  Total unique papers: {len(all_papers)}")
    print(f"  With open-access PDF: {with_pdf}")
    print(f"  Downloaded: {downloaded}")
    return downloaded


# ---------------------------------------------------------------------------
# Mode: process — 未処理PDFの数式抽出
# ---------------------------------------------------------------------------

def run_process_pdfs() -> int:
    """未処理PDFから数式を抽出。build_equation_graph.py のロジックを利用。"""
    print("=" * 60)
    print("PROCESS UNPROCESSED PDFs")
    print("=" * 60)

    from build_equation_graph import (
        load_processed_pdfs,
        save_processed_pdfs,
        merge_and_dedupe,
    )
    from extract_equations import extract_equations

    data = load_unified(UNIFIED_JSON)
    total_before = len(data)
    processed = load_processed_pdfs(PROCESSED_PDFS_FILE)

    all_pdfs = sorted(INPUT_PDF_DIR.glob("*.pdf"))
    unprocessed = [p for p in all_pdfs if p.name not in processed]

    print(f"Total PDFs: {len(all_pdfs)}, Already processed: {len(processed)}, To process: {len(unprocessed)}")

    if not unprocessed:
        print("No unprocessed PDFs.")
        return 0

    added_total = 0
    for i, pdf_path in enumerate(unprocessed):
        print(f"\n[{i+1}/{len(unprocessed)}] {pdf_path.name}")
        try:
            equations = extract_equations(pdf_path)
            before = len(data)
            data = merge_and_dedupe(data, equations)
            added = len(data) - before
            added_total += added
            print(f"  Extracted {len(equations)} equations, {added} new")

            save_unified(data, UNIFIED_JSON)
            processed.add(pdf_path.name)
            save_processed_pdfs(PROCESSED_PDFS_FILE, processed)

        except Exception as e:
            print(f"  ERROR: {e}")

        if i < len(unprocessed) - 1:
            print(f"  Sleeping 15s (5 RPM)...")
            time.sleep(15)

    print(f"\nPDF processing done: {total_before} → {len(data)} equations (+{added_total})")
    return added_total


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="データセット拡大 (975式 → 3,000+式)")
    parser.add_argument(
        "--mode",
        choices=["handbook", "handbook2", "papers", "process", "split-large", "all"],
        default="all",
        help="実行モード",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="API呼び出しなしで計画のみ表示",
    )
    args = parser.parse_args()

    data = load_unified(UNIFIED_JSON)
    print(f"Current dataset: {len(data)} equations\n")

    if args.mode in ("handbook", "all"):
        run_handbook_expansion(dry_run=args.dry_run)

    if args.mode in ("handbook2", "all"):
        run_handbook_expansion_r2(dry_run=args.dry_run)

    if args.mode in ("papers", "all"):
        run_paper_search(dry_run=args.dry_run)

    if args.mode in ("split-large", "all"):
        run_split_large_pdfs(dry_run=args.dry_run)

    if args.mode in ("process", "all"):
        if not args.dry_run:
            run_process_pdfs()
        else:
            print("\n[DRY RUN] Would process unprocessed PDFs")

    # 最終サマリー
    data = load_unified(UNIFIED_JSON)
    print(f"\n{'='*60}")
    print(f"FINAL: {len(data)} equations in {UNIFIED_JSON.name}")
    from collections import Counter
    sources = Counter(d["source_id"] for d in data)
    print(f"Sources: {len(sources)}")
    for s, c in sources.most_common(10):
        print(f"  {c:4d}  {s}")


if __name__ == "__main__":
    main()
