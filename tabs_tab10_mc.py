from dataclasses import dataclass
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import hashlib, json
import pandas as pd
from typing import Literal, Dict, Tuple

@dataclass
class Params:
    T: float = 313.15
    Vaq: float = 0.33
    Vorg: float = 0.40
    # cinética
    k1f: float = 2.0e-2
    k1r: float = 1.0e-3
    k2:  float = 1.0e-2
    # aperturas en orgánica
    k_FA:  float = 2.0e-4   # Ep + 2*FA
    k_PFA: float = 1.0e-4   # Ep + 2*PFA
    # transferencia (dos-películas)
    kla_H2O2: float = 1.0e-3
    kla_HCOOH: float = 1.0e-3
    kla_PFA:  float = 1.0e-3
    kla_H2O:  float = 1.0e-3
    # partición (C_org = Kp * C_aq)
    Kp_H2O2: float = 0.02
    Kp_HCOOH: float = 0.20
    Kp_PFA:  float = 0.20
    Kp_H2O:  float = 1.0
    # actividades
    activities: Literal["IDEAL","UNIQUAC","UNIFAC"] = "IDEAL"

def conc(n: float, V: float) -> float:
    return n / max(V, 1e-12)

MW_H2O2  = 34.0147
MW_HCOOH = 46.0254
MW_H2O   = 18.0153
R = 8.314462618 # J/mol/K

def _safe_uid():
    # tomá tu UID de donde lo guardes; si no hay, usa "local"
    return st.session_state.get("uid") or "local"

def _save_last(db, data):
    if db is None:
        raise RuntimeError("No hay handle 'db' para Firestore.")
    uid = _safe_uid()
    db.collection("users").document(uid)\
      .collection("mc_scenarios").document("ultimo").set(data)

def _collect_params(inputs):
    # 'inputs' es un dict con TODAS las variables UI
    return {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in inputs.items()}

def _apply_params_to_widgets(d):
    # Devuelve un diccionario con defaults si faltan claves
    defs = _defaults()
    defs.update(d or {})
    return defs

def _defaults():
    return dict(
        # Composición Mi PoliOL (volúmenes por lote)
        V_soy=400.00, V_H2SO4=3.64, V_H2O=32.73, V_HCOOH=80.00, V_H2O2=204.36,
        # Cinética
        k1f=2.0e-2, k1r=1.0e-3, k2=1.0e-2, k3=1.0e-4, k4=2.0e-5, k5=5.0e-5, alpha=1.0,
        # Transferencia de masa (PFA, H2O2, HCOOH, H2O)
        frac_aq=0.25,
        kla_PFA=5e-3,  Kp_PFA=5.0,
        kla_H2O2=1e-3, Kp_H2O2=0.05,
        kla_HCOOH=3e-3, Kp_HCOOH=0.20,
        kla_H2O=3e-3,  Kp_H2O=0.02,
        # Simulación
        t_h=12.0, npts=400
    )

@dataclass
class P:
    k1f: float; k1r: float; k2: float; k3: float; k4: float; k5: float; alpha: float
    Vaq: float; Vorg: float
    kla_PFA: float; kla_H2O2: float; Kp_PFA: float; Kp_H2O2: float
    kla_HCOOH: float = 0.0
    Kp_HCOOH: float  = 0.20
    kla_H2O: float   = 0.0
    Kp_H2O: float    = 0.02

def _params_hash(prm: dict) -> str:
    # Solo cosas que cambian la simulación (no sliders de ejes, ni selección de series)
    keys = ["V_soy","V_H2SO4","V_H2O","V_HCOOH","V_H2O2",
            "k1f","k1r","k2","k3","k4","k5","alpha",
            "frac_aq","kla_PFA","Kp_PFA","kla_H2O2","Kp_H2O2",
            "kla_HCOOH","Kp_HCOOH","kla_H2O","Kp_H2O",
            "t_h","npts"]
    subset = {k: float(prm[k]) for k in keys}
    s = json.dumps(subset, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

# ==== Cinética  ====
K_FIXED = dict(
    k1f=2.0e-2,  # L·mol⁻¹·s⁻¹
    k1r=1.0e-3,  # s⁻¹
    k2 =1.0e-2,  # L·mol⁻¹·s⁻¹
    k3 =1.0e-4,  # s⁻¹
    k4 =2.0e-5,  # s⁻¹
    k5 =5.0e-5,  # L·mol⁻¹·s⁻¹
    alpha=1.0    # –
)

# Unidades 
K_META = {
    "k1f":  {"unid":"L·mol⁻¹·s⁻¹", "det":"(definir)"},
    "k1r":  {"unid":"s⁻¹",          "det":"(definir)"},
    "k2":   {"unid":"L·mol⁻¹·s⁻¹", "det":"(definir)"},
    "k3":   {"unid":"s⁻¹",          "det":"(definir)"},
    "k4":   {"unid":"s⁻¹",          "det":"(definir)"},
    "k5":   {"unid":"L·mol⁻¹·s⁻¹", "det":"(definir)"},
    "alpha":{"unid":"–",            "det":"(definir)"},
}

def rhs_one_phase(t, y, p: Params):
    # y = [n_CdC, n_Ep, n_FA, n_PFA, n_H2O2, n_HCOOH]  (todo en un volumen efectivo V=Vorg+Vaq)
    n_CdC, n_Ep, n_FA, n_PFA, n_H2O2, n_HCOOH = y
    V = p.Vorg + p.Vaq
    C_CdC, C_Ep, C_FA, C_PFA, C_H2O2, C_HCOOH = [conc(n, V) for n in y]
    r1f = p.k1f * C_H2O2 * C_HCOOH
    r1r = p.k1r * C_PFA
    r_epox = p.k2 * C_PFA * C_CdC
    r_open_FA  = p.k_FA  * C_Ep * (C_FA**2)
    r_open_PFA = p.k_PFA * C_Ep * (C_PFA**2)
    dn_CdC = - r_epox * V
    dn_Ep  = ( r_epox - r_open_FA - r_open_PFA ) * V
    dn_FA  = (-r1f + r1r + r_epox) * V
    dn_PFA = ( r1f - r1r - r_epox - r_open_PFA ) * V
    dn_H2O2  = - r1f * V
    dn_HCOOH = (-r1f + r1r + r_epox) * V
    return [dn_CdC, dn_Ep, dn_FA, dn_PFA, dn_H2O2, dn_HCOOH]

def rhs_two_phase_eq(t,y,p:Params):
    n_CdC,n_Ep,n_FAo,n_PFAo,n_H2O2a,n_HCOOHa = y
    CCo,Ep,FAo,PFAo = [conc(v,p.Vorg) for v in [n_CdC,n_Ep,n_FAo,n_PFAo]]
    H2O2a,HCOOHa    = [conc(v,p.Vaq)  for v in [n_H2O2a,n_HCOOHa]]
    PFAa = PFAo / max(p.Kp_PFA,   1e-12)

    r1f = p.k1f*H2O2a*HCOOHa
    r1r = p.k1r*PFAa
    r_epox     = p.k2*PFAo*CCo
    r_open_FA  = p.k_FA  * Ep * (FAo**2)
    r_open_PFA = p.k_PFA * Ep * (PFAo**2)

    # ¡OJO con los volúmenes!
    dn_CdC   = - r_epox * p.Vorg
    dn_Ep    = ( r_epox - r_open_FA - r_open_PFA ) * p.Vorg
    dn_FAo   = (-r1f + r1r) * p.Vaq + r_epox * p.Vorg
    dn_PFAo  = ( r1f - r1r) * p.Vaq - r_epox * p.Vorg - r_open_PFA * p.Vorg
    dn_H2O2a = - r1f * p.Vaq
    dn_HCOOHa= (-r1f + r1r) * p.Vaq + r_epox * p.Vorg
    return [dn_CdC,dn_Ep,dn_FAo,dn_PFAo,dn_H2O2a,dn_HCOOHa]

def rhs_two_phase_twofilm(t, y, p: Params):
    # y = [n_CdC, n_Ep, n_FAo, n_PFAo, n_H2O2a, n_HCOOHa, n_PFAa, n_FAa]
    n_CdC, n_Ep, n_FAo, n_PFAo, n_H2O2a, n_HCOOHa, n_PFAa, n_FAa = y
    C_CdC = conc(n_CdC, p.Vorg); C_Ep = conc(n_Ep, p.Vorg)
    C_FAo = conc(n_FAo, p.Vorg); C_PFAo = conc(n_PFAo, p.Vorg)
    C_H2O2a = conc(n_H2O2a, p.Vaq); C_HCOOHa = conc(n_HCOOHa, p.Vaq)
    C_PFAa = conc(n_PFAa, p.Vaq);   C_FAa   = conc(n_FAa, p.Vaq)
    # transferencia  (C_org* = Kp * C_aq)
    C_PFAo_star = p.Kp_PFA  * C_PFAa
    C_FAo_star  = p.Kp_HCOOH* C_FAa
    J_PFA = p.kla_PFA  * (C_PFAo_star - C_PFAo) * p.Vorg
    J_FA  = p.kla_HCOOH* (C_FAo_star  - C_FAo ) * p.Vorg
    # reacciones
    r1f = p.k1f * C_H2O2a * C_HCOOHa
    r1r = p.k1r * C_PFAa
    r_epox = p.k2 * C_PFAo * C_CdC
    r_open_FA  = p.k_FA  * C_Ep * (C_FAo**2)
    r_open_PFA = p.k_PFA * C_Ep * (C_PFAo**2)
    # balances en MOLES
    dn_CdC = - r_epox * p.Vorg
    dn_Ep  = ( r_epox - r_open_FA - r_open_PFA ) * p.Vorg
    dn_PFAo= - r_epox*p.Vorg - r_open_PFA*p.Vorg + J_PFA
    dn_FAo = + r_epox*p.Vorg + J_FA
    dn_H2O2a  = - r1f*p.Vaq
    dn_HCOOHa = (-r1f + r1r)*p.Vaq
    dn_PFAa   = ( r1f - r1r)*p.Vaq - J_PFA
    dn_FAa    = (-r1f + r1r)*p.Vaq - J_FA
    return [dn_CdC, dn_Ep, dn_FAo, dn_PFAo, dn_H2O2a, dn_HCOOHa, dn_PFAa, dn_FAa]


def simulate_models(p: Params, y0: Dict[str, float], t_span: Tuple[float, float], npts: int = 400):
    t_eval = np.linspace(t_span[0], t_span[1], npts)
    # 1 fase
    y01 = [y0[k] for k in ["CdC","Ep","FA","PFA","H2O2","HCOOH"]]
    sol1 = solve_ivp(lambda t,y: rhs_one_phase(t,y,p), t_span, y01, t_eval=t_eval, method="LSODA")
    # 2 fases eq
    y02 = [y0[k] for k in ["CdC","Ep","FAo","PFAo","H2O2a","HCOOHa"]]
    sol2 = solve_ivp(lambda t,y: rhs_two_phase_eq(t,y,p), t_span, y02, t_eval=t_eval, method="LSODA")
    # 2 fases 2 films
    y03 = [y0[k] for k in ["CdC","Ep","FAo","PFAo","H2O2a","HCOOHa","PFAa","FAa"]]
    sol3 = solve_ivp(lambda t,y: rhs_two_phase_twofilm(t,y,p), t_span, y03, t_eval=t_eval, method="LSODA")
    return {"t": t_eval, "1F": sol1.y, "2F_eq": sol2.y, "2F_2film": sol3.y}

def render_tab10(db=None, mostrar_sector_flotante=lambda *a, **k: None):
    if "mc_params" not in st.session_state:
        st.session_state["mc_params"] = _defaults()
    prm = _apply_params_to_widgets(st.session_state["mc_params"])
    prm.update({k: float(v) for k, v in K_FIXED.items()})

    def _fmt_e(x): return f"{x:.2e}"
    k = K_FIXED  # o prm si preferís
    colw = [5, 2]

    # R1
    c1, c2 = st.columns(colw)
    with c1:
        st.latex(r"\mathrm{HCOOH + H_2O_2 \xrightleftharpoons[k_{1r}]{k_{1f}} PFA + H_2O}\tag{R1}")
    with c2:
        st.markdown(
            f"""
            <div style="text-align:right; font-size:0.9em; margin-top:0.6rem">
            k₁f = {_fmt_e(k['k1f'])} L·mol⁻¹·s⁻¹<br>
            k₁r = {_fmt_e(k['k1r'])} s⁻¹
            </div>
            """, unsafe_allow_html=True
        )

    # R2
    c1, c2 = st.columns(colw)
    with c1:
        st.latex(r"\mathrm{PFA + C{=}C \xrightarrow{k_{2}} Ep + HCOOH}\tag{R2}")
    with c2:
        st.markdown(
            f"<div style='text-align:right; font-size:0.9em; margin-top:1.2rem'>k₂ = {_fmt_e(k['k2'])} L·mol⁻¹·s⁻¹</div>",
            unsafe_allow_html=True
        )

    # R3
    c1, c2 = st.columns(colw)
    with c1:
        st.latex(r"\mathrm{PFA \xrightarrow{k_{3}} HCOOH}\tag{R3}")
    with c2:
        st.markdown(
            f"<div style='text-align:right; font-size:0.9em; margin-top:1.2rem'>k₃ = {_fmt_e(k['k3'])} s⁻¹</div>",
            unsafe_allow_html=True
        )

    # R4
    c1, c2 = st.columns(colw)
    with c1:
        st.latex(r"\mathrm{H_2O_2 \xrightarrow{k_{4}} H_2O}\tag{R4}")
    with c2:
        st.markdown(
            f"<div style='text-align:right; font-size:0.9em; margin-top:1.2rem'>k₄ = {_fmt_e(k['k4'])} s⁻¹</div>",
            unsafe_allow_html=True
        )

    # R5
    c1, c2 = st.columns(colw)
    with c1:
        st.latex(r"\mathrm{Ep + H_2O \xrightarrow{k_{5}} Open}\tag{R5}")
    with c2:
        st.markdown(
            f"<div style='text-align:right; font-size:0.9em; margin-top:1.2rem'>k₅ = {_fmt_e(k['k5'])} L·mol⁻¹·s⁻¹</div>",
            unsafe_allow_html=True
        )
        
    # ---- α separado, alineado con la columna de los k ----
    _ , right = st.columns(colw)  # misma geometría
    with right:
        st.markdown(
            f"<div style='text-align:right; font-size:0.9em; margin-top:0.0rem'>"
            f"α = {k['alpha']:.2f} factor ácido en R1, R2, R5."
            f"</div>",  # ← cerrar
            unsafe_allow_html=True
        )

    # Referencias
    st.markdown("""
    **Referencias**
    - R1: Formación del ácido perfórmico  
    - R2: Epoxidación en fase orgánica  
    - R3: Descomposición del PFA  
    - R4: Descomposición del H₂O₂  
    - R5: Apertura del epóxido
    """)

    st.markdown("**Modelo 1-fase**")
    st.latex(r"""
    \begin{aligned}
    \dot C_{H_2O_2} &= -k_{1f} C_{HCOOH} C_{H_2O_2}\,\alpha + k_{1r} C_{PFA} - k_4 C_{H_2O_2}\\
    \dot C_{HCOOH} &= -k_{1f} C_{HCOOH} C_{H_2O_2}\,\alpha + k_{1r} C_{PFA} + k_2 C_{PFA} C_{C{=}C}\,\alpha + k_3 C_{PFA}\\
    \dot C_{PFA}   &= \phantom{-}k_{1f} C_{HCOOH} C_{H_2O_2}\,\alpha - k_{1r} C_{PFA} - k_2 C_{PFA} C_{C{=}C}\,\alpha - k_3 C_{PFA}\\
    \dot C_{C{=}C} &= -k_{2} C_{PFA} C_{C{=}C}\,\alpha\\
    \dot C_{Ep}    &= \phantom{-}k_{2} C_{PFA} C_{C{=}C}\,\alpha - k_{5} C_{Ep} C_{H_2O}\,\alpha\\
    \dot C_{Open}  &= \phantom{-}k_{5} C_{Ep} C_{H_2O}\,\alpha\\
    \dot C_{H_2O}  &= \phantom{-}k_{1r} C_{PFA} + k_{4} C_{H_2O_2}
    \end{aligned}
    """)

    # ——— Ecuaciones de balance (modelo 2-fases) ———
    st.markdown("**Modelo 2-fases**")
    st.latex(r"\frac{dC_{H_2O_2,aq}}{dt} = -\,k_{1f}\,C_{HCOOH,aq}\,C_{H_2O_2,aq} + k_{1r}\,C_{PFA,aq} - k_{4}\,C_{H_2O_2,aq} \;-\; \frac{\dot n_{H_2O_2}^{TM}}{V_{aq}}\tag{R6}")
    st.latex(r"\frac{dC_{H_2O_2,org}}{dt} = +\,\frac{\dot n_{H_2O_2}^{TM}}{V_{org}} - k_{4}\,C_{H_2O_2,org}\tag{R7}")

    st.latex(r"\frac{dC_{HCOOH,aq}}{dt} = -\,k_{1f}\,C_{HCOOH,aq}\,C_{H_2O_2,aq} + k_{1r}\,C_{PFA,aq} + k_{3}\,C_{PFA,aq} \;-\; \frac{\dot n_{HCOOH}^{TM}}{V_{aq}}\tag{R8}")
    st.latex(r"\frac{dC_{HCOOH,org}}{dt} = +\,\frac{\dot n_{HCOOH}^{TM}}{V_{org}}\tag{R9}")

    st.latex(r"\frac{dC_{PFA,aq}}{dt} = +\,k_{1f}\,C_{HCOOH,aq}\,C_{H_2O_2,aq} - k_{1r}\,C_{PFA,aq} - k_{3}\,C_{PFA,aq} \;-\; \frac{\dot n_{PFA}^{TM}}{V_{aq}}\tag{R10}")
    st.latex(r"\frac{dC_{PFA,org}}{dt} = -\,k_{2}\,C_{PFA,org}\,C_{C{=}C,org} \;+\; \frac{\dot n_{PFA}^{TM}}{V_{org}}\tag{R11}")

    st.latex(r"\frac{dC_{C{=}C,org}}{dt} = -\,k_{2}\,C_{PFA,org}\,C_{C{=}C,org}\tag{R12}")
    st.latex(r"\frac{dC_{Ep,org}}{dt} = +\,k_{2}\,C_{PFA,org}\,C_{C{=}C,org} - k_{5}\,C_{Ep,org}\,C_{H_2O,org}\tag{R13}")

    st.latex(r"\frac{dC_{H_2O,aq}}{dt} = +\,k_{1r}\,C_{PFA,aq} + k_{4}\,C_{H_2O_2,aq} \;-\; \frac{\dot n_{H_2O}^{TM}}{V_{aq}}\tag{R14}")
    st.latex(r"\frac{dC_{H_2O,org}}{dt} = +\,\frac{\dot n_{H_2O}^{TM}}{V_{org}} \;-\; k_{5}\,C_{Ep,org}\,C_{H_2O,org}\tag{R15}")

    # ======================= UI: IMPORTAR JSON ===============================
    st.markdown("**Importar parámetros (JSON)**")
    up = st.file_uploader("Cargar JSON de escenario", type=["json"])
    if "mc_params" not in st.session_state:
        st.session_state["mc_params"] = _defaults()
    if up is not None:
        try:
            loaded = json.load(up)
            st.session_state["mc_params"] = _apply_params_to_widgets(loaded)
            st.success("Parámetros cargados desde JSON.")
        except Exception as e:
            st.error(f"JSON inválido: {e}")

    prm = _apply_params_to_widgets(st.session_state["mc_params"])

    # === Constantes físico-químicas (fijas) ===
    densidades = {
        "ACEITE": 0.910,  # Aceite de soja crudo [g/mL]
        "H2SO4":  1.83,   # Ácido sulfúrico 98% [g/mL]
        "H2O":    1.00,   # Agua [g/mL]
        "HCOOH":  1.195,  # Ácido fórmico 85% [g/mL]
        "H2O2":   1.11,   # Peróxido 30% p/v [g/mL]
        "HCOOOH": 1.18,   # Ácido perfórmico [g/mL]
    }

    MW = {
        "ACEITE":  873.64,
        "H2SO4":    98.08,
        "H2O":      18.02,
        "HCOOH":    46.03,
        "H2O2":     34.01,
        "HCOOOH":   62.02,
    }

    # (opcional) Aliases si en el resto del archivo usás variables sueltas:
    MW_H2O2  = MW["H2O2"]
    MW_H2O   = MW["H2O"]
    MW_HCOOH = MW["HCOOH"]


    # ================== Composición inicial (con %v/v, %p/p, d, PM, moles y equivalentes) ================== 
    st.markdown("**Composición inicial**")

    # Ingreso de aceite (los demás se escalan por la receta base)
    V_soy_in = st.number_input("Aceite de soja crudo [mL]", min_value=0.0,
                            value=float(prm.get("V_soy", 400.0)), step=0.1, format="%.4f")

    ratios_vs_oil = {"H2SO4":3.64/400.0, "H2O":32.73/400.0, "HCOOH":80.0/400.0, "H2O2":204.36/400.0}
    V_H2SO4 = ratios_vs_oil["H2SO4"] * V_soy_in
    V_H2O   = ratios_vs_oil["H2O"]   * V_soy_in
    V_HCOOH = ratios_vs_oil["HCOOH"] * V_soy_in
    V_H2O2  = ratios_vs_oil["H2O2"]  * V_soy_in

    # --- Fracciones por PRIORIDAD en el aceite (orgánico) ---
    V_total_mix = V_soy_in + V_H2SO4 + V_H2O + V_HCOOH + V_H2O2
    frac_org = V_soy_in / max(V_total_mix, 1e-12)
    frac_aq_calc = 1.0 - frac_org

    # Volúmenes de fase (L)
    V_total_L = max(V_total_mix / 1000.0, 1e-12)
    Vaq  = max(frac_aq_calc * V_total_L, 1e-12)
    Vorg = max(V_total_L - Vaq, 1e-12)

    prm["frac_aq"] = float(frac_aq_calc)

    # Actualizar prm para el resto de la app
    prm["V_soy"], prm["V_H2SO4"], prm["V_H2O"], prm["V_HCOOH"], prm["V_H2O2"] = \
        float(V_soy_in), float(V_H2SO4), float(V_H2O), float(V_HCOOH), float(V_H2O2)

    # Moles (según definición de cada solución)
    n_soy   = (densidades["ACEITE"] * V_soy_in) / MW["ACEITE"]
    n_H2SO4 = (0.98 * densidades["H2SO4"] * V_H2SO4) / MW["H2SO4"]
    n_H2O   = (densidades["H2O"]   * V_H2O)   / MW["H2O"]
    n_HCOOH = (0.85 * densidades["HCOOH"] * V_HCOOH) / MW["HCOOH"]
    g_H2O2  = 0.30 * V_H2O2                         # 30 g H2O2 / 100 mL
    n_H2O2  = g_H2O2 / MW["H2O2"]

    # Equivalentes (mol eq) – factores corregidos
    eq_soy   = 4.5 * n_soy     # C=C por mol aceite
    eq_H2SO4 = 2.0 * n_H2SO4   # 2 H+
    eq_H2O   = 1.0 * n_H2O     # 1 nucleófilo
    eq_HCOOH = 1.0 * n_HCOOH   # 1 H+
    eq_H2O2  = 1.0 * n_H2O2    # 1 oxidante

    # Masas para % p/p
    m_soy   = densidades["ACEITE"] * V_soy_in
    m_H2SO4 = densidades["H2SO4"]  * V_H2SO4
    m_H2O   = densidades["H2O"]    * V_H2O
    m_HCOOH = densidades["HCOOH"]  * V_HCOOH
    m_H2O2  = densidades["H2O2"]   * V_H2O2
    m_total = max(m_soy + m_H2SO4 + m_H2O + m_HCOOH + m_H2O2, 1e-12)

    datos = [
        ("Aceite de soja crudo",  V_soy_in, densidades["ACEITE"], MW["ACEITE"], n_soy,   eq_soy,   m_soy,   None, None),
        ("Ácido sulfúrico 98%",   V_H2SO4,  densidades["H2SO4"],  MW["H2SO4"],  n_H2SO4, eq_H2SO4, m_H2SO4, None, None),
        ("Agua destilada",        V_H2O,    densidades["H2O"],    MW["H2O"],    n_H2O,   eq_H2O,   m_H2O,   prm.get("Kp_H2O"),   prm.get("kla_H2O")),
        ("Ácido fórmico 85%",     V_HCOOH,  densidades["HCOOH"],  MW["HCOOH"],  n_HCOOH, eq_HCOOH, m_HCOOH, prm.get("Kp_HCOOH"), prm.get("kla_HCOOH")),
        ("Peróxido H₂O₂ 30% p/v", V_H2O2,   densidades["H2O2"],   MW["H2O2"],   n_H2O2,  eq_H2O2,  m_H2O2,  prm.get("Kp_H2O2"),  prm.get("kla_H2O2")),
        ("PFA (generado)",        0.0,      densidades["HCOOOH"], MW["HCOOOH"],    0.0,     0.0,     0.0,   prm.get("Kp_PFA"),   prm.get("kla_PFA")),
        ]

    df_comp = pd.DataFrame(datos, columns=["Componente","Volumen [mL]","d [g/mL]","PM [g/mol]","n [mol]","n eq [mol eq]","m [g]","Koq","kLa [1/s]"])

    # % v/v y % p/p
    V_total = max(df_comp["Volumen [mL]"].sum(), 1e-12)
    df_comp["[% v/v]"] = 100.0 * df_comp["Volumen [mL]"] / V_total
    df_comp["[% p/p]"] = 100.0 * df_comp["m [g]"] / m_total

    # Orden de columnas
    df_comp = df_comp[["Componente","[% v/v]","[% p/p]","Volumen [mL]","d [g/mL]","PM [g/mol]","n [mol]","n eq [mol eq]","Koq","kLa [1/s]"]]

    st.dataframe(
        df_comp.style.format({
            "[% v/v]": "{:.4f}",
            "[% p/p]": "{:.4f}",
            "Volumen [mL]": "{:.4f}",
            "d [g/mL]": "{:.4f}",
            "PM [g/mol]": "{:.4f}",
            "n [mol]": "{:.4f}",
            "n eq [mol eq]": "{:.4f}",
            "Koq": "{:.2f}",
            "kLa [1/s]": "{:.2e}",
        }, na_rep=""),
        use_container_width=True, hide_index=True
    )

    # Observación en letra más pequeña
    st.markdown(
        "<p style='font-size: 0.7em;'>*Koq: coeficiente de partición o coeficiente de reparto. <br> "
        "**kLa: coeficiente volumétrico de transferencia de masa gas–líquido o líquido–líquido.</p>",
        unsafe_allow_html=True
    )


    # ======================= Relaciones molares (sin recálculos) =======================
    st.markdown("**Relaciones molares**")

    # Agua de reactivos (usa m_* y g_H2O2 ya calculados arriba)
    n_H2O_from_H2SO4 = (0.02 * m_H2SO4) / MW["H2O"]              # 2% m/m en H2SO4 98%
    n_H2O_from_HCOOH = (0.15 * m_HCOOH) / MW["H2O"]              # 15% m/m en HCOOH 85%
    n_H2O_from_H2O2  = max(m_H2O2 - g_H2O2, 0.0) / MW["H2O"]     # resto en la solución 30% p/v
    n_H2O_react      = n_H2O_from_H2SO4 + n_H2O_from_HCOOH + n_H2O_from_H2O2

    # Agua total = agua de reactivos + agua dosificada (n_H2O ya calculado arriba)
    n_H2O_total = n_H2O_react + n_H2O

    # Protones (equivalentes, ya tenés n_H2SO4 y n_HCOOH)
    Hplus_H2SO4 = 2.0 * n_H2SO4
    Hplus_HCOOH = 1.0 * n_HCOOH
    Hplus_total = Hplus_H2SO4 + Hplus_HCOOH

    # Dobles enlaces (ya tenés eq_soy = 4.5*n_soy)
    n_CdC = eq_soy

    # Relaciones (reusan n_* existentes)
    rel_H2O2_CdC  = n_H2O2 / n_CdC if n_CdC > 0 else 0.0
    rel_H2SO4_CdC = n_H2SO4 / n_CdC if n_CdC > 0 else 0.0
    rel_HCOOH_CdC = n_HCOOH / n_CdC if n_CdC > 0 else 0.0
    rel_H2O2_soy  = n_H2O2 / n_soy if n_soy > 0 else 0.0
    rel_H2SO4_soy = n_H2SO4 / n_soy if n_soy > 0 else 0.0
    rel_HCOOH_soy = n_HCOOH / n_soy if n_soy > 0 else 0.0

    fmt = "{:.4f}"
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.dataframe(
            pd.DataFrame([
                ("C=C (dobles enlaces) [mol]",  n_CdC),
                ("Agua de reactivos [mol]",     n_H2O_react),
                ("Agua total [mol]",            n_H2O_total),
                ("Fracción orgánica Vorg/V [–]", 1.0 - prm["frac_aq"]),
            ], columns=["Magnitud", "Valor"]).style.format({"Valor": fmt}),
            use_container_width=True, hide_index=True
        )

    with col2:
        st.dataframe(
            pd.DataFrame([
                ("Protones H₂SO₄ [mol H⁺]",   Hplus_H2SO4),
                ("Protones HCOOH [mol H⁺]",   Hplus_HCOOH),
                ("Protones totales [mol H⁺]", Hplus_total),
            ], columns=["Magnitud", "Valor"]).style.format({"Valor": fmt}),
            use_container_width=True, hide_index=True
        )

    with col3:
        st.dataframe(
            pd.DataFrame([
                ("Relación H₂O₂/C=C [mol/mol]",  rel_H2O2_CdC),
                ("Relación H₂SO₄/C=C [mol/mol]", rel_H2SO4_CdC),
                ("Relación HCOOH/C=C [mol/mol]", rel_HCOOH_CdC),
            ], columns=["Magnitud", "Valor"]).style.format({"Valor": fmt}),
            use_container_width=True, hide_index=True
        )

    with col4:
        st.dataframe(
            pd.DataFrame([
                ("Relación H₂O₂/aceite [mol/mol]",  rel_H2O2_soy),
                ("Relación H₂SO₄/aceite [mol/mol]", rel_H2SO4_soy),
                ("Relación HCOOH/aceite [mol/mol]", rel_HCOOH_soy),
            ], columns=["Magnitud", "Valor"]).style.format({"Valor": fmt}),
            use_container_width=True, hide_index=True
        )
    # ================================================================================

    # ======================= Simulación =====================================
    s1, s2 = st.columns(2)
    prm["t_h"]  = s1.number_input("Tiempo total [h]", value=prm["t_h"], step=0.5)
    prm["npts"] = s2.number_input("Puntos", value=int(prm["npts"]), step=50, min_value=100)

    # Guardar en estado para exportación
    st.session_state["mc_params"] = prm

    # ===== y0 para los 3 modelos EN MOLES (convertimos desde tus y0_* en concentración) =====
    t_end  = float(prm["t_h"])*3600.0
    npts   = int(prm["npts"])

    # ---- definir y0_1fase y V_total_L ----
    V_total_L = Vaq + Vorg

    # Estado inicial 2-fases en CONCENTRACIÓN  (aq primero, org después)
    # Usa moles que ya calculaste: n_H2O2, n_HCOOH, n_CdC (=eq_soy), n_H2O_total
    y0_2fases = [
        n_H2O2 / Vaq,          # Ca_H2O2
        n_HCOOH / Vaq,         # Ca_HCOOH
        0.0,                   # Ca_PFA
        0.0,                   # Co_H2O2 (inicialmente 0 en orgánico)
        0.0,                   # Co_HCOOH
        0.0,                   # Co_PFA
        n_CdC / Vorg,          # Co_CdC (dobles enlaces en orgánico)
        0.0,                   # Co_Ep
        0.0,                   # Co_Open
        n_H2O_total / Vaq,     # Ca_H2O  (agua total en acuosa a t=0)
        0.0                    # Co_H2O
    ]

    # y0_2fases = [Ca_H2O2, Ca_HCOOH, Ca_PFA, Co_H2O2, Co_HCOOH, Co_PFA, Co_CdC, Co_Ep, Co_Open, Ca_H2O, Co_H2O]
    Ca_H2O2, Ca_HCOOH, Ca_PFA, Co_H2O2, Co_HCOOH, Co_PFA, Co_CdC, Co_Ep, Co_Open, Ca_H2O, Co_H2O = y0_2fases

    # Para el modelo 1F (pseudo-homogéneo) definimos una mezcla “equivalente” en el volumen total.
    # Tomamos especies principalmente “acuosas” desde la fase aq y “orgánicas” desde org:
    H2O2_1F = Ca_H2O2   # mol/L
    HCOOH_1F = (Ca_HCOOH*Vaq + Co_HCOOH*Vorg) / max(V_total_L, 1e-12)  # promedio ponderado
    PFA_1F  = (Ca_PFA*Vaq + Co_PFA*Vorg) / max(V_total_L, 1e-12)
    CdC_1F  = Co_CdC     # mol/L (está en org)
    Ep_1F   = Co_Ep
    Open_1F = Co_Open

    y0_1fase = [H2O2_1F, HCOOH_1F, PFA_1F, CdC_1F, Ep_1F, Open_1F]


    # 1F (usa V_total_L)   y0_1fase = [H2O2, HCOOH, PFA, C=C, Ep, Open, H2O]  (concentraciones)
    y0_1F_moles = {
        "H2O2":  y0_1fase[0]*V_total_L,
        "HCOOH": y0_1fase[1]*V_total_L,
        "PFA":   y0_1fase[2]*V_total_L,
        "CdC":   y0_1fase[3]*V_total_L,
        "Ep":    y0_1fase[4]*V_total_L,
        "FA":    y0_1fase[1]*V_total_L  # FA total en 1F: usamos HCOOH inicial
    }

    # 2F eq / 2F 2films (usa Vaq, Vorg)
    # y0_2fases = [
    #   0 Ca_H2O2, 1 Ca_HCOOH, 2 Ca_PFA,
    #   3 Co_H2O2, 4 Co_HCOOH, 5 Co_PFA,
    #   6 Co_CdC,  7 Co_Ep,    8 Co_Open,
    #   9 Ca_H2O,  10 Co_H2O
    # ]
    y0 = {
        # clave 1F
        "CdC":  y0_1F_moles["CdC"],
        "Ep":   y0_1F_moles["Ep"],
        "FA":   y0_1F_moles["FA"],
        "PFA":  y0_1F_moles["PFA"],
        "H2O2": y0_1F_moles["H2O2"],
        "HCOOH":y0_1F_moles["HCOOH"],
        # clave 2F-eq / 2F-2films
        "FAo":   y0_2fases[4]*Vorg,   # HCOOH org inicial
        "PFAo":  y0_2fases[5]*Vorg,   # PFA org inicial
        "H2O2a": y0_2fases[0]*Vaq,    # H2O2 acuosa
        "HCOOHa":y0_2fases[1]*Vaq,    # HCOOH acuosa
        "PFAa":  y0_2fases[2]*Vaq,    # PFA acuosa
        "FAa":   y0_2fases[1]*Vaq     # FA acuosa (= HCOOH aq)
    }

    # Parámetros del modelo (usa sliders/JSON de prm donde corresponda)
    par_2fases = Params(
        T = 313.15,
        Vaq = Vaq, Vorg = Vorg,
        k1f = prm["k1f"], k1r = prm["k1r"], k2 = prm["k2"],
        # aperturas en orgánica (puedes exponer sliders luego si querés)
        k_FA = 2.0e-4,
        k_PFA = 1.0e-4,
        # transferencia y partición desde prm
        kla_PFA = prm["kla_PFA"],  Kp_PFA  = prm["Kp_PFA"],
        kla_H2O2 = prm["kla_H2O2"], Kp_H2O2 = prm["Kp_H2O2"],
        kla_HCOOH = prm["kla_HCOOH"], Kp_HCOOH = prm["Kp_HCOOH"],
        kla_H2O = prm["kla_H2O"],   Kp_H2O = prm["Kp_H2O"],
    )

    # ===== Ejecutar los 3 modelos (usa las RHS nuevas en moles) =====
    res = simulate_models(par_2fases, y0, (0, t_end), npts=npts)

    # Unidad de graficación
    colu1 = st.columns([1.6])
    unidad = colu1[0].radio("Unidad", ["Moles de lote", "Concentración (mol/L)"], index=0, horizontal=True)
    if unidad == "Moles de lote":
        conv_1F   = lambda arr: arr * V_total_L
        conv_2F_aq  = lambda arr: arr * Vaq
        conv_2F_org = lambda arr: arr * Vorg
        ylab = "Cantidad (mol)"
    else:
        conv_1F   = lambda arr: arr / max(V_total_L,1e-12) * V_total_L  # identidad para mantener consistencia
        conv_2F_aq  = lambda arr: arr / max(Vaq,1e-12) * Vaq
        conv_2F_org = lambda arr: arr / max(Vorg,1e-12) * Vorg
        ylab = "Concentración (mol/L)"

    times_h = res["t"]/3600.0


    def _add_traces(fig, t, Ys, names, conv_fn):
        for y, name in zip(Ys, names):
            fig.add_trace(go.Scatter(x=t, y=conv_fn(y), mode="lines", name=name))
        fig.update_layout(xaxis_title="Tiempo [h]", yaxis_title=ylab,
                        legend_title="Especie", hovermode="x unified")

    times_h = res["t"]/3600.0

    # ---- Modelo 1 fase (1F) ----
    names_1F = ["C=C","Ep","FA","PFA","H2O2","HCOOH"]   # índices 0..5 en res["1F"]
    Ys_1F = [res["1F"][i] for i in range(6)]
    fig1 = go.Figure()
    _add_traces(fig1, times_h, Ys_1F, names_1F, conv_1F)
    fig1.update_layout(title="Modelo 1-fase – Todas las especies")
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Modelo 2 fases – equilibrio ----
    # org: 0..3  |  aq: 4..5   en res["2F_eq"]
    names_2Feq_org = ["C=C(org)","Ep(org)","FA(org)","PFA(org)"]
    names_2Feq_aq  = ["H2O2(aq)","HCOOH(aq)"]
    Ys_2Feq_org = [res["2F_eq"][i] for i in [0,1,2,3]]
    Ys_2Feq_aq  = [res["2F_eq"][i] for i in [4,5]]
    fig2 = go.Figure()
    _add_traces(fig2, times_h, Ys_2Feq_org, names_2Feq_org, conv_2F_org)
    _add_traces(fig2, times_h, Ys_2Feq_aq,  names_2Feq_aq,  conv_2F_aq)
    fig2.update_layout(title="Modelo 2-fases (equilibrio) – Todas las especies")
    st.plotly_chart(fig2, use_container_width=True)

    # ---- Modelo 2 fases – dos películas ----
    # org: 0..3  |  aq: 4..7   en res["2F_2film"]
    names_2Ftf_org = ["C=C(org)","Ep(org)","FA(org)","PFA(org)"]
    names_2Ftf_aq  = ["H2O2(aq)","HCOOH(aq)","PFA(aq)","FA(aq)"]
    Ys_2Ftf_org = [res["2F_2film"][i] for i in [0,1,2,3]]
    Ys_2Ftf_aq  = [res["2F_2film"][i] for i in [4,5,6,7]]
    fig3 = go.Figure()
    _add_traces(fig3, times_h, Ys_2Ftf_org, names_2Ftf_org, conv_2F_org)
    _add_traces(fig3, times_h, Ys_2Ftf_aq,  names_2Ftf_aq,  conv_2F_aq)
    fig3.update_layout(title="Modelo 2-fases (dos películas) – Todas las especies")
    st.plotly_chart(fig3, use_container_width=True)



    def pack_for_plots(res):
        t=res["t"]
        return {
            "t": t,
            "1F":    {"CdC":res["1F"][0],    "Ep":res["1F"][1],    "FA":res["1F"][2],    "PFA":res["1F"][3]},
            "2F_eq": {"CdC":res["2F_eq"][0], "Ep":res["2F_eq"][1], "FA":res["2F_eq"][2], "PFA":res["2F_eq"][3]},
            "2F_2film":{"CdC":res["2F_2film"][0],"Ep":res["2F_2film"][1],"FA":res["2F_2film"][2],"PFA":res["2F_2film"][3]},
        }

    # Pie: simplificaciones
    st.markdown("""
---
**Simplificaciones:** isoterma, volumen constante, cinética de orden potencia, α catalítico lumped; en 2-fases, \\(k_L a\\) y \\(K_{oq}\\) constantes.
""")
