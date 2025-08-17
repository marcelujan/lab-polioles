import json
from dataclasses import dataclass
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
#  Pestaña: Modelo cinético – Mi PoliOL (explícito + persistencia + JSON I/O)
# ─────────────────────────────────────────────────────────────────────────────

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
        # Composición Mi PoliOL (volúmenes por lote)
        V_soy=400.00, V_H2SO4=3.64, V_H2O=32.73, V_HCOOH=80.00, V_H2O2=204.36,
        moles_CdC=2.00,
        # Densidades
        rho_soy=0.92, rho_HCOOH=1.215, rho_H2O2=1.00,
        # Cinética
        k1f=2.0e-2, k1r=1.0e-3, k2=1.0e-2, k3=1.0e-4, k4=2.0e-5, k5=5.0e-5, alpha=1.0,
        # Transferencia de masa
        usar_TM=False, frac_aq=0.25, kla_PFA=5e-3, Kp_PFA=5.0, kla_H2O2=1e-3, Kp_H2O2=0.05,
        # Simulación
        t_h=4.0, npts=400
    )

@dataclass
class P:
    k1f: float; k1r: float; k2: float; k3: float; k4: float; k5: float; alpha: float
    Vaq: float; Vorg: float; kla_PFA: float; kla_H2O2: float; Kp_PFA: float; Kp_H2O2: float
    usar_TM: bool

def render_tab10(db=None, mostrar_sector_flotante=lambda *a, **k: None):
    st.title("Modelo cinético – Mi PoliOL")
    st.caption("Ecuaciones explícitas, 1-fase o 2-fases con transferencia de masa, guardado automático en Firebase y exportación/importación JSON.")

    # ======================= ECUACIONES EXPLÍCITAS ============================
    st.markdown(r"""
### Esquema y ecuaciones
**(R1)** Formación perácido (PFA):  
\[
\mathrm{HCOOH + H_2O_2 \xrightleftharpoons[k_{1r}]{k_{1f}} PFA + H_2O}
\]
**(R2)** Epoxidación (org): \(\mathrm{PFA + C{=}C \xrightarrow{k_2} Ep + HCOOH}\)  
**(R3)** Decaimiento del PFA: \(\mathrm{PFA \xrightarrow{k_3} HCOOH}\)  
**(R4)** Decaimiento del H_2O_2: \(\mathrm{H_2O_2 \xrightarrow{k_4} H_2O}\)  
**(R5)** Apertura del epóxido: \(\mathrm{Ep + H_2O \xrightarrow{k_5} Open}\)

**Modelo 1-fase (concentraciones \(C\) en mol·L⁻¹):**
\[
\begin{aligned}
\dot C_{H_2O_2} &= -k_{1f} C_{HCOOH}C_{H_2O_2}\alpha + k_{1r}C_{PFA} - k_4 C_{H_2O_2}\\
\dot C_{HCOOH}&= -k_{1f} C_{HCOOH}C_{H_2O_2}\alpha + k_{1r}C_{PFA} + k_2 C_{PFA}C_{C=C}\alpha + k_3 C_{PFA}\\
\dot C_{PFA}   &=  k_{1f} C_{HCOOH}C_{H_2O_2}\alpha - k_{1r}C_{PFA} - k_2 C_{PFA}C_{C=C}\alpha - k_3 C_{PFA}\\
\dot C_{C=C}   &= -k_2 C_{PFA}C_{C=C}\alpha\\
\dot C_{Ep}    &=  k_2 C_{PFA}C_{C=C}\alpha - k_5 C_{Ep}C_{H_2O}\alpha\\
\dot C_{Open}  &=  k_5 C_{Ep}C_{H_2O}\alpha\\
\dot C_{H_2O}  &=  k_{1r}C_{PFA} + k_4 C_{H_2O_2}
\end{aligned}
\]

**Transferencia de masa (opcional, dos fases):**  
Fase acuosa (aq) y orgánica (org), \(V=V_{aq}+V_{org}\). Para \(i \in \{PFA,H_2O_2\}\):  
\[
\dot n_i^{TM} = k_L a\left(C_{i,aq} - \frac{C_{i,org}}{K_{oq}}\right)
\]
Se suman términos \(+\,\dot n_i^{TM}/V_{aq}\) en aq y \( -\,\dot n_i^{TM}/V_{org}\) en org.
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
    st.subheader("Transferencia de masa (opcional)")
    prm["usar_TM"] = st.checkbox("Activar dos fases con TM", value=prm["usar_TM"])
    if prm["usar_TM"]:
        t1,t2,t3 = st.columns(3)
        prm["frac_aq"]  = t1.slider("Fracción acuosa Vaq/V", 0.05, 0.60, value=float(prm["frac_aq"]), step=0.05)
        prm["kla_PFA"]  = t2.number_input("kLa_PFA [1/s]", value=prm["kla_PFA"], format="%.2e")
        prm["Kp_PFA"]   = t2.number_input("Koq PFA (=Corg/Caq)", value=prm["Kp_PFA"], step=0.5)
        prm["kla_H2O2"] = t3.number_input("kLa_H2O2 [1/s]", value=prm["kla_H2O2"], format="%.2e")
        prm["Kp_H2O2"]  = t3.number_input("Koq H2O2", value=prm["Kp_H2O2"], step=0.01)

    # ======================= UI: TIEMPO =====================================
    st.subheader("Simulación")
    s1, s2 = st.columns(2)
    prm["t_h"]  = s1.number_input("Tiempo total [h]", value=prm["t_h"], step=0.5)
    prm["npts"] = s2.number_input("Puntos", value=int(prm["npts"]), step=50, min_value=100)

    # Guardar en estado para exportación
    st.session_state["mc_params"] = prm

    # =============== CÁLCULOS INICIALES (moles y estados) ===================
    g_H2O2   = 0.30 * prm["V_H2O2"]                        # p/v
    n_H2O2   = g_H2O2 / MW_H2O2
    g_HCOOH  = 0.85 * prm["rho_HCOOH"] * prm["V_HCOOH"]
    n_HCOOH  = g_HCOOH / MW_HCOOH
    g_H2O_ini= (prm["V_H2O"]*1.0) + 0.15*(prm["rho_HCOOH"]*prm["V_HCOOH"]) + \
               max(prm["rho_H2O2"]*prm["V_H2O2"] - g_H2O2, 0.0)
    n_H2O    = g_H2O_ini / MW_H2O
    V_total_L= (prm["V_soy"] + prm["V_H2SO4"] + prm["V_H2O"] + prm["V_HCOOH"] + prm["V_H2O2"])/1000.0

    if prm["usar_TM"]:
        Vaq  = float(prm["frac_aq"])*V_total_L
        Vorg = V_total_L - Vaq
        y0 = np.array([
            n_H2O2/Vaq,               # Ca_H2O2
            n_HCOOH/Vaq,              # Ca_HCOOH
            0.0,                      # Ca_PFA
            0.0,                      # Co_PFA
            prm["moles_CdC"]/Vorg,    # Co_CdC
            0.0, 0.0,                 # Co_Ep, Co_Open
            n_H2O/Vaq                 # Ca_H2O
        ])
        par = P(prm["k1f"],prm["k1r"],prm["k2"],prm["k3"],prm["k4"],prm["k5"],prm["alpha"],
                Vaq,Vorg, prm["kla_PFA"],prm["kla_H2O2"],prm["Kp_PFA"],prm["Kp_H2O2"], True)
    else:
        y0 = np.array([
            n_H2O2/V_total_L,
            n_HCOOH/V_total_L,
            0.0,
            prm["moles_CdC"]/V_total_L,
            0.0, 0.0,
            n_H2O/V_total_L
        ])
        par = P(prm["k1f"],prm["k1r"],prm["k2"],prm["k3"],prm["k4"],prm["k5"],prm["alpha"],
                V_total_L,0.0, 0.0,0.0,5.0,0.05, False)

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
        Ca_H2O2, Ca_HCOOH, Ca_PFA, Co_PFA, Co_CdC, Co_Ep, Co_Open, Ca_H2O = y
        r1f = p.k1f*Ca_HCOOH*Ca_H2O2*p.alpha; r1r=p.k1r*Ca_PFA; r3=p.k3*Ca_PFA; r4=p.k4*Ca_H2O2
        r2  = p.k2*Co_PFA*Co_CdC*p.alpha;     r5=p.k5*Co_Ep*Ca_H2O*p.alpha
        TM_PFA  = p.kla_PFA*(Ca_PFA - Co_PFA/p.Kp_PFA)
        TM_H2O2 = p.kla_H2O2*(Ca_H2O2 - 0.0/p.Kp_H2O2)
        return [
            -r1f + r1r - r4 - TM_H2O2,     # dCa_H2O2
            -r1f + r1r + r3,               # dCa_HCOOH
             r1f - r1r - r3 - TM_PFA,      # dCa_PFA
             TM_PFA - r2,                  # dCo_PFA
            -r2,                           # dCo_CdC
             r2 - r5,                      # dCo_Ep
             r5,                           # dCo_Open
             r1r + r4                      # dCa_H2O
        ]

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

    # Simulación
    if run_clicked:
        t_end = float(prm["t_h"])*3600.0
        t_eval= np.linspace(0, t_end, int(prm["npts"]))
        if prm["usar_TM"]:
            sol = solve_ivp(lambda t,Y: rhs_2phase(t,Y,par), [0,t_end], y0, t_eval=t_eval,
                            method="LSODA", rtol=1e-7, atol=1e-9)
            def aqmol(c):  return c*par.Vaq
            def orgmol(c): return c*par.Vorg

            plt.figure()
            plt.plot(sol.t/3600, aqmol(sol.y[0]), label="H₂O₂ (aq)")
            plt.plot(sol.t/3600, aqmol(sol.y[2]), label="PFA (aq)")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles"); plt.title("Oxidantes – fase acuosa")
            plt.legend(); st.pyplot(plt.gcf())

            plt.figure()
            plt.plot(sol.t/3600, orgmol(sol.y[3]), label="PFA (org)")
            plt.plot(sol.t/3600, orgmol(sol.y[4]), label="C=C (org)")
            plt.plot(sol.t/3600, orgmol(sol.y[5]), label="Epóxido (org)")
            plt.plot(sol.t/3600, orgmol(sol.y[6]), label="Apertura (org)")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles"); plt.title("Orgánico – PFA, C=C, Ep, Open")
            plt.legend(); st.pyplot(plt.gcf())

            st.json({
                "H2O2_aq_final_mol": float(aqmol(sol.y[0,-1])),
                "PFA_aq_final_mol": float(aqmol(sol.y[2,-1])),
                "PFA_org_final_mol": float(orgmol(sol.y[3,-1])),
                "Epox_org_final_mol": float(orgmol(sol.y[5,-1])),
                "Open_org_final_mol": float(orgmol(sol.y[6,-1]))
            })

        else:
            sol = solve_ivp(lambda t,Y: rhs_1phase(t,Y,par), [0,t_end], y0, t_eval=t_eval,
                            method="LSODA", rtol=1e-7, atol=1e-9)

            def mol(c): return c*par.Vaq  # en 1-fase usamos V_total en par.Vaq
            plt.figure()
            plt.plot(sol.t/3600, mol(sol.y[0]), label="H₂O₂")
            plt.plot(sol.t/3600, mol(sol.y[2]), label="PFA")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles"); plt.title("Oxidantes – 1 fase")
            plt.legend(); st.pyplot(plt.gcf())

            plt.figure()
            plt.plot(sol.t/3600, mol(sol.y[3]), label="C=C")
            plt.plot(sol.t/3600, mol(sol.y[4]), label="Epóxido")
            plt.plot(sol.t/3600, mol(sol.y[5]), label="Apertura")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles"); plt.title("C=C / Epóxido / Apertura – 1 fase")
            plt.legend(); st.pyplot(plt.gcf())

            st.json({
                "H2O2_final_mol":  float(mol(sol.y[0,-1])),
                "PFA_final_mol":   float(mol(sol.y[2,-1])),
                "Epox_final_mol":  float(mol(sol.y[4,-1])),
                "Open_final_mol":  float(mol(sol.y[5,-1]))
            })

    # Pie: simplificaciones
    st.markdown("""
---
**Simplificaciones:** isoterma, volumen constante, cinética de orden potencia, α catalítico lumped; en 2-fases, \(k_L a\) y \(K_{oq}\) constantes.
""")

    # Botón flotante (si lo tenés)
    try:
        mostrar_sector_flotante(st.session_state.get("db"), key_suffix="mc")
    except Exception:
        pass
