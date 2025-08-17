import json
from dataclasses import dataclass
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

MW_H2O2  = 34.0147
MW_HCOOH = 46.0254
MW_H2O   = 18.0153

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
        # Composición Mi PoliOL
        V_soy=400.00, V_H2SO4=3.64, V_H2O=32.73, V_HCOOH=80.00, V_H2O2=204.36,
        moles_CdC=2.00,
        # Densidades
        rho_soy=0.92, rho_HCOOH=1.215, rho_H2O2=1.00,
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

def render_tab10(db=None, mostrar_sector_flotante=lambda *a, **k: None):
    # ───────── Esquema y ecuaciones (render LaTeX) ─────────
    st.subheader("Esquema y ecuaciones")

    st.latex(r"\mathrm{HCOOH + H_2O_2 \xrightleftharpoons[k_{1r}]{k_{1f}} PFA + H_2O}\tag{R1}")
    st.latex(r"\mathrm{PFA + C{=}C \xrightarrow{k_{2}} Ep + HCOOH}\tag{R2}")
    st.latex(r"\mathrm{PFA \xrightarrow{k_{3}} HCOOH}\tag{R3}")
    st.latex(r"\mathrm{H_2O_2 \xrightarrow{k_{4}} H_2O}\tag{R4}")
    st.latex(r"\mathrm{Ep + H_2O \xrightarrow{k_{5}} Open}\tag{R5}")

    st.markdown("""
    **Referencias**  
    **(R1)** Formación del ácido perfórmico (PFA) / retroceso.  
    **(R2)** Epoxidación en fase orgánica.  
    **(R3)** Decaimiento del PFA.  
    **(R4)** Descomposición del H₂O₂.  
    **(R5)** Apertura del epóxido por agua.
    """)

    st.markdown("**Modelo 1-fase (concentraciones \(C\) en mol·L⁻¹)**")
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

    # ───────── Bloque: TM y balances 2-fases con ecuaciones numeradas ─────────
    # ——— Ecuaciones de balance (modelo 2-fases) ———
    st.markdown("### Ecuaciones de balance (modelo 2-fases)")
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

    st.markdown(r"""
    **Referencia (modelo 2-fases)**  
    (R6–R7): H₂O₂ en acuosa / orgánica  
    (R8–R9): HCOOH en acuosa / orgánica  
    (R10–R11): PFA en acuosa / orgánica  
    (R12): C=C en orgánica  
    (R13): Epóxido en orgánica  
    (R14–R15): Agua en acuosa / orgánica  

    **Signo de TM:** \( \dot n_i^{TM} > 0 \Rightarrow \) flujo **aq → org** (− en aq, + en org).
    """)

    # ======================= UI: IMPORTAR JSON ===============================
    st.subheader("Importar parámetros (JSON)")
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

    # ======================= UI: COMPOSICIÓN ================================
    st.subheader("Composición inicial (volúmenes por lote)")
    c1, c2, c3 = st.columns(3)
    with c1:
        prm["V_soy"]   = st.number_input("Aceite de soja [mL]",  value=prm["V_soy"], step=1.0)
        prm["V_H2SO4"] = st.number_input("H₂SO₄ 98% [mL]",       value=prm["V_H2SO4"], step=0.1)
        prm["V_H2O"]   = st.number_input("Agua destilada [mL]",  value=prm["V_H2O"], step=0.1)
    with c2:
        prm["V_HCOOH"] = st.number_input("HCOOH 85% [mL]",       value=prm["V_HCOOH"], step=0.1)
        prm["V_H2O2"]  = st.number_input("H₂O₂ 30% **p/v** [mL]",value=prm["V_H2O2"], step=0.1,
                                         help="30 g H2O2 / 100 mL de solución (p/v)")
        prm["moles_CdC"]=st.number_input("Moles C=C (lote)",     value=prm["moles_CdC"], step=0.1)
    with c3:
        st.markdown("**Densidades:**")
        prm["rho_soy"]   = st.number_input("ρ aceite [g/mL]",    value=prm["rho_soy"], step=0.01)
        prm["rho_HCOOH"] = st.number_input("ρ HCOOH 85% [g/mL]", value=prm["rho_HCOOH"], step=0.005)
        prm["rho_H2O2"]  = st.number_input("ρ H₂O₂ 30% p/v [g/mL]", value=prm["rho_H2O2"], step=0.01)

    # ======================= UI: CINÉTICA ===================================
    st.subheader("Constantes cinéticas y factor ácido")
    kcols = st.columns(7)
    keys = ["k1f","k1r","k2","k3","k4","k5","alpha"]
    labels= ["k1f [L/mol/s]","k1r [1/s]","k2 [L/mol/s]","k3 [1/s]","k4 [1/s]","k5 [L/mol/s]","α (ácido)"]
    fmts  = ["%.2e"]*6+["%0.2f"]
    for i,(k,lab,fmt) in enumerate(zip(keys,labels,fmts)):
        prm[k] = kcols[i].number_input(lab, value=prm[k], format=fmt)

    # ======================= UI: TRANSFERENCIA DE MASA ======================
    st.subheader("Parámetros de transferencia de masa")
    with st.expander("Ajustes de TM (modelo 2-fases)"):
        cols_tm1, cols_tm2 = st.columns(2)
        with cols_tm1:
            prm["frac_aq"]    = st.slider("Fracción acuosa Vaq/V", 0.05, 0.60, value=float(prm["frac_aq"]), step=0.05)
            prm["kla_PFA"]    = st.number_input("kLa_PFA [1/s]",   value=prm["kla_PFA"],   format="%.2e")
            prm["Kp_PFA"]     = st.number_input("Koq PFA (=Corg/Caq)", value=prm["Kp_PFA"], step=0.1)
            prm["kla_HCOOH"]  = st.number_input("kLa_HCOOH [1/s]", value=prm["kla_HCOOH"], format="%.2e")
            prm["Kp_HCOOH"]   = st.number_input("Koq HCOOH",       value=prm["Kp_HCOOH"],  step=0.01)
        with cols_tm2:
            prm["kla_H2O2"]   = st.number_input("kLa H₂O₂ [1/s]",  value=prm["kla_H2O2"],  format="%.2e")
            prm["Kp_H2O2"]    = st.number_input("Koq H₂O₂",        value=prm["Kp_H2O2"],   step=0.01)
            prm["kla_H2O"]    = st.number_input("kLa H₂O [1/s]",   value=prm["kla_H2O"],   format="%.2e")  # NUEVO
            prm["Kp_H2O"]     = st.number_input("Koq H₂O",         value=prm["Kp_H2O"],    step=0.01)      # NUEVO

    # ======================= UI: TIEMPO =====================================
    st.subheader("Simulación")
    s1, s2 = st.columns(2)
    prm["t_h"]  = s1.number_input("Tiempo total [h]", value=prm["t_h"], step=0.5)
    prm["npts"] = s2.number_input("Puntos", value=int(prm["npts"]), step=50, min_value=100)

    # Guardar en estado para exportación
    st.session_state["mc_params"] = prm

    # =============== CÁLCULOS INICIALES (moles y estados) ===================
    g_H2O2    = 0.30 * prm["V_H2O2"]              # p/v: 30 g por 100 mL
    n_H2O2    = g_H2O2 / MW_H2O2
    g_HCOOH   = 0.85 * prm["rho_HCOOH"] * prm["V_HCOOH"]
    n_HCOOH   = g_HCOOH / MW_HCOOH
    g_H2O_ini = (prm["V_H2O"]*1.0) + 0.15*(prm["rho_HCOOH"]*prm["V_HCOOH"]) + \
                max(prm["rho_H2O2"]*prm["V_H2O2"] - g_H2O2, 0.0)
    n_H2O     = g_H2O_ini / MW_H2O

    V_total_L = (prm["V_soy"] + prm["V_H2SO4"] + prm["V_H2O"] + prm["V_HCOOH"] + prm["V_H2O2"]) / 1000.0
    # 2-fases: usar frac_aq para partición de volumen
    Vaq       = float(prm["frac_aq"]) * V_total_L
    Vorg      = V_total_L - Vaq

    # ---- MODELO 1-FASE (7 vars) ----
    # y1 = [H2O2, HCOOH, PFA, C=C, Ep, Open, H2O]
    y0_1fase = np.array([
        n_H2O2 / V_total_L,
        n_HCOOH / V_total_L,
        0.0,
        prm["moles_CdC"] / V_total_L,
        0.0, 0.0,
        n_H2O / V_total_L
    ])
    par_1fase = P(
        prm["k1f"], prm["k1r"], prm["k2"], prm["k3"], prm["k4"], prm["k5"], prm["alpha"],
        V_total_L, 0.0,
        0.0, 0.0, 5.0, 0.05,
        kla_HCOOH=0.0, Kp_HCOOH=0.20, kla_H2O=0.0, Kp_H2O=0.02
    )

    # ---- MODELO 2-FASES (11 vars) ----
    # Orden en y2 y en rhs_2phase:
    # [0] Ca_H2O2, [1] Ca_HCOOH, [2] Ca_PFA,
    # [3] Co_H2O2, [4] Co_HCOOH, [5] Co_PFA,
    # [6] Co_CdC,  [7] Co_Ep,    [8] Co_Open,
    # [9] Ca_H2O,  [10] Co_H2O   << NUEVO
    y0_2fases = np.array([
        n_H2O2 / Vaq,            # Ca_H2O2
        n_HCOOH / Vaq,           # Ca_HCOOH
        0.0,                     # Ca_PFA
        0.0,                     # Co_H2O2
        0.0,                     # Co_HCOOH
        0.0,                     # Co_PFA
        prm["moles_CdC"] / Vorg, # Co_CdC
        0.0, 0.0,                # Co_Ep, Co_Open
        n_H2O / Vaq,             # Ca_H2O
        0.0                      # Co_H2O
    ])
    par_2fases = P(
        prm["k1f"], prm["k1r"], prm["k2"], prm["k3"], prm["k4"], prm["k5"], prm["alpha"],
        Vaq, Vorg,
        prm["kla_PFA"], prm["kla_H2O2"], prm["Kp_PFA"], prm["Kp_H2O2"],
        kla_HCOOH=prm["kla_HCOOH"], Kp_HCOOH=prm["Kp_HCOOH"],
        kla_H2O=prm["kla_H2O"],     Kp_H2O=prm["Kp_H2O"]
    )

    # ========================= RHS (1-fase y 2-fases) =======================
    def rhs_1phase(t, y, p: P):
        H2O2, HCOOH, PFA, CdC, Ep, Open, H2O = y
        r1f = p.k1f*HCOOH*H2O2*p.alpha; r1r=p.k1r*PFA
        r2  = p.k2*PFA*CdC*p.alpha;     r3 = p.k3*PFA
        r4  = p.k4*H2O2;                r5 = p.k5*Ep*H2O*p.alpha
        return [
            -r1f + r1r - r4,
            -r1f + r1r + r2 + r3,
             r1f - r1r - r2 - r3,
            -r2,
             r2 - r5,
             r5,
             r1r + r4
        ]

    def rhs_2phase(t, y, p: P):
        (Ca_H2O2, Ca_HCOOH, Ca_PFA,
        Co_H2O2, Co_HCOOH, Co_PFA,
        Co_CdC,  Co_Ep,    Co_Open,
        Ca_H2O,  Co_H2O) = y

        # Reacciones en acuosa
        r1f = p.k1f*Ca_HCOOH*Ca_H2O2*p.alpha
        r1r = p.k1r*Ca_PFA
        r3  = p.k3*Ca_PFA
        r4a = p.k4*Ca_H2O2              # descomp. H2O2 (aq)

        # Reacciones en orgánica
        r2  = p.k2*Co_PFA*Co_CdC*p.alpha
        r5  = p.k5*Co_Ep*Co_H2O*p.alpha  # << AHORA usa agua orgánica
        r4o = p.k4*Co_H2O2               # descomp. H2O2 (org)

        # Transferencias (dos-películas)
        TM_H2O2  = p.kla_H2O2 *(Ca_H2O2 - Co_H2O2 / p.Kp_H2O2)
        TM_HCOOH = p.kla_HCOOH*(Ca_HCOOH - Co_HCOOH/ p.Kp_HCOOH)
        TM_PFA   = p.kla_PFA  *(Ca_PFA   - Co_PFA   / p.Kp_PFA)
        TM_H2O   = p.kla_H2O  *(Ca_H2O   - Co_H2O   / p.Kp_H2O)  # << NUEVO

        # Balances
        dCa_H2O2  = -r1f + r1r - r4a - TM_H2O2
        dCa_HCOOH = -r1f + r1r + r3  - TM_HCOOH
        dCa_PFA   =  r1f - r1r - r3  - TM_PFA

        dCo_H2O2  = +TM_H2O2 - r4o
        dCo_HCOOH = +TM_HCOOH
        dCo_PFA   = +TM_PFA - r2

        dCo_CdC   = -r2
        dCo_Ep    =  r2 - r5
        dCo_Open  =  r5

        dCa_H2O   =  r1r + r4a - TM_H2O          # agua formada en aq y que puede transferirse
        dCo_H2O   = +TM_H2O - r5                 # agua que llega a org y reacciona en R5

        return [dCa_H2O2, dCa_HCOOH, dCa_PFA,
                dCo_H2O2, dCo_HCOOH, dCo_PFA,
                dCo_CdC,  dCo_Ep,    dCo_Open,
                dCa_H2O,  dCo_H2O]


    # ========================= BOTONES: SIM, GUARDAR, EXPORT =================
    cbtn = st.columns([1,1,1,1.2])
    run_clicked  = cbtn[0].button("▶ Ejecutar")
    save_clicked = cbtn[1].button("💾 Guardar ‘último’ (Firebase)")
    export_clicked = cbtn[2].button("📤 Exportar JSON")
    reset_clicked  = cbtn[3].button("↺ Reset a valores por defecto")

    # Export JSON
    if export_clicked:
        js = json.dumps(_collect_params(prm), indent=2)
        st.download_button("Descargar parámetros (JSON)", js, file_name="mc_params.json", mime="application/json")

    if reset_clicked:
        st.session_state["mc_params"] = _defaults()
        st.experimental_rerun()

    # Guardar “último” en Firestore
    if save_clicked:
        try:
            _save_last(db, _collect_params(prm))
            st.success("Guardado en Firebase: users/{uid}/mc_scenarios/ultimo")
        except Exception as e:
            st.warning(f"No se pudo guardar en Firestore: {e}")

    def _plot_all_one_figure(times_h, curves, title, y_label):
        fig = go.Figure()
        for name, y in curves.items():
            fig.add_trace(go.Scatter(x=times_h, y=y, mode="lines", name=name))
        fig.update_layout(title=title, xaxis_title="Tiempo [h]", yaxis_title=y_label,
                        legend_title="Especie", hovermode="x unified")
        return fig

    def _apply_axes(fig, auto_axes, x_min, x_max, y_min, y_max):
        if not auto_axes:
            fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(range=[y_min, y_max])
        return fig

    # Simulación
    if run_clicked:
        # 1) Resolver ODEs (siempre ambos modelos)
        t_end  = float(prm["t_h"])*3600.0
        t_eval = np.linspace(0, t_end, int(prm["npts"]))

        sol1 = solve_ivp(lambda t,Y: rhs_1phase(t,Y,par_1fase), [0,t_end], y0_1fase, t_eval=t_eval,
                        method="LSODA", rtol=1e-7, atol=1e-9)
        sol2 = solve_ivp(lambda t,Y: rhs_2phase(t,Y,par_2fases), [0,t_end], y0_2fases, t_eval=t_eval,
                        method="LSODA", rtol=1e-7, atol=1e-9)

        times_h = sol1.t/3600.0  # mismo grid para ambos

        # 2) Controles de visualización (siempre presentes)
        st.subheader("Visualización")
        colu1 = st.columns([1.6])
        unidad = colu1[0].radio("Unidad", ["Moles de lote", "Concentración (mol/L)"], index=0, horizontal=True)

        # Conversión según unidad
        if unidad == "Moles de lote":
            conv1    = lambda c: c*par_1fase.Vaq
            conv_aq  = lambda c: c*par_2fases.Vaq
            conv_org = lambda c: c*par_2fases.Vorg
            ylab = "Cantidad (moles)"
        else:
            conv1 = conv_aq = conv_org = (lambda c: c)
            ylab = "Concentración (mol/L)"

        # Curvas convertidas
        curves1 = {
            "H₂O₂":     conv1(sol1.y[0]),
            "PFA":      conv1(sol1.y[2]),
            "C=C":      conv1(sol1.y[3]),
            "Epóxido":  conv1(sol1.y[4]),
            "Apertura": conv1(sol1.y[5]),
        }
        curves2 = {
            "H₂O₂ (aq)":      conv_aq(sol2.y[0]),
            "H₂O₂ (org)":     conv_org(sol2.y[3]),
            "HCOOH (aq)":     conv_aq(sol2.y[1]),
            "HCOOH (org)":    conv_org(sol2.y[4]),
            "PFA (aq)":       conv_aq(sol2.y[2]),
            "PFA (org)":      conv_org(sol2.y[5]),
            "C=C (org)":      conv_org(sol2.y[6]),
            "Epóxido (org)":  conv_org(sol2.y[7]),
            "Apertura (org)": conv_org(sol2.y[8]),
            "H₂O (aq)":       conv_aq(sol2.y[9]),
            "H₂O (org)":      conv_org(sol2.y[10]),
        }

        # Filtros de series (primero se elige, luego auto-limites)
        all_keys_1 = list(curves1.keys())
        sel_keys_1 = st.multiselect("Series (Modelo 1-fase)", options=all_keys_1, default=all_keys_1, key="sel_1fase")

        all_keys_2 = list(curves2.keys())
        sel_keys_2 = st.multiselect("Series (Modelo 2-fases)", options=all_keys_2, default=all_keys_2, key="sel_2fases")

        curves1_f = {k: v for k, v in curves1.items() if k in sel_keys_1}
        curves2_f = {k: v for k, v in curves2.items() if k in sel_keys_2}

        # Auto-límites a partir de las series seleccionadas (ignorando H2O si hay otras)
        pool = {**curves1_f, **curves2_f}
        pool_sin_agua = {k: v for k, v in pool.items() if "H₂O" not in k}
        if pool_sin_agua:
            pool = pool_sin_agua

        if pool:
            y_max_auto = float(max(np.nanmax(np.asarray(y)) for y in pool.values()))
            y_min_auto = float(min(np.nanmin(np.asarray(y)) for y in pool.values()))
        else:
            y_min_auto, y_max_auto = 0.0, 1.0

        # pequeño padding
        pad = 0.05 * (y_max_auto - y_min_auto if y_max_auto > y_min_auto else 1.0)
        t_end_h = float(prm["t_h"])

        # ── 4 boxes SIEMPRE visibles, inicializados con autos ──
        t_min_h, t_max_h = 0.0, float(prm["t_h"])
        cax1, cax2, cax3, cax4 = st.columns(4)
        x_min = cax1.number_input("x min [h]", value=t_min_h, step=0.5)
        x_max = cax2.number_input("x max [h]", value=t_max_h, step=0.5)
        y_min = cax3.number_input("y min", value=max(0.0, y_min_auto - pad))
        y_max = cax4.number_input("y max", value=y_max_auto + pad)

        # ── Saneamiento de rangos para que nunca colapsen ──
        EPSX, EPSY = 1e-6, 1e-12

        # clamp dentro del dominio simulado
        x_min = max(t_min_h, min(x_min, t_max_h - EPSX))
        x_max = max(x_min + EPSX, min(x_max, t_max_h))

        # evitar y-span ≤ 0 y NaN/inf
        def _isfinite(v): 
            return np.isfinite(v) and not np.isnan(v)

        if not _isfinite(y_min): y_min = 0.0
        if not _isfinite(y_max): y_max = y_min + 1.0
        if y_max <= y_min + EPSY:
            y_max = y_min + max(EPSY, 0.05*max(1.0, abs(y_min)))

        # ── Graficar ──
        fig1 = _plot_all_one_figure(sol1.t/3600.0, curves1_f, "Modelo 1-fase", ylab)
        fig2 = _plot_all_one_figure(sol2.t/3600.0, curves2_f, "Modelo 2-fases (con TM)", ylab)

        # aplicar SIEMPRE los límites saneados
        fig1.update_xaxes(range=[x_min, x_max]); fig1.update_yaxes(range=[y_min, y_max])
        fig2.update_xaxes(range=[x_min, x_max]); fig2.update_yaxes(range=[y_min, y_max])

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)


    # Pie: simplificaciones
    st.markdown("""
---
**Simplificaciones:** isoterma, volumen constante, cinética de orden potencia, α catalítico lumped; en 2-fases, \(k_L a\) y \(K_{oq}\) constantes.
""")
