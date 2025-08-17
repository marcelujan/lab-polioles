import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# ---------------------------
#  Pestaña: Modelo cinético
# ---------------------------

def render_tab10(db, mostrar_sector_flotante):
    st.title("Modelo cinético – Mi PoliOL")
    st.session_state["current_tab"] = "Modelo cinético"

    # --- Panel: Composición inicial (volúmenes por lote) ---
    st.subheader("Composición inicial (Mi PoliOL)")
    colA, colB, colC = st.columns(3)
    with colA:
        V_soy = st.number_input("Aceite de soja crudo [mL]", value=400.00, step=1.0)
        V_H2SO4 = st.number_input("Ácido sulfúrico 98% [mL]", value=3.64, step=0.1)
        V_H2O = st.number_input("Agua destilada [mL]", value=32.73, step=0.1)
    with colB:
        V_HCOOH = st.number_input("Ácido fórmico 85% [mL]", value=80.00, step=0.1)
        V_H2O2 = st.number_input("Peróxido 30% p/v [mL]", value=204.36, step=0.1)
        mol_CdC = st.number_input("Moles iniciales de C=C (lote)", value=2.00, step=0.1,
                                   help="Estimación a partir del IY/oxirano esperado")
    with colC:
        st.markdown("**Densidades (editar si querés exactitud):**")
        rho_soy   = st.number_input("ρ aceite [g/mL]", value=0.92, step=0.01)
        rho_H2SO4 = st.number_input("ρ H₂SO₄ 98% [g/mL]", value=1.84, step=0.01)
        rho_HCOOH = st.number_input("ρ HCOOH 85% [g/mL]", value=1.215, step=0.005)
        rho_H2O2  = st.number_input("ρ H₂O₂ 30% p/v [g/mL]", value=1.00, step=0.01)
        rho_H2Oli = st.number_input("ρ mezcla orgánica [g/mL] (p/volumen total)", value=0.95, step=0.01)

    # --- Panel: Parámetros cinéticos ---
    st.subheader("Constantes cinéticas (editar)")
    with st.expander("Perácido y epoxidación"):
        col1, col2, col3 = st.columns(3)
        with col1:
            k1f = st.number_input("k1f (HCOOH+H₂O₂→PFA) [L/mol/s]", value=2.0e-2, format="%.2e")
            k1r = st.number_input("k1r (PFA→HCOOH+H₂O₂) [1/s]", value=1.0e-3, format="%.2e")
        with col2:
            k2  = st.number_input("k2 (PFA+C=C→Epóxido+HCOOH) [L/mol/s]", value=1.0e-2, format="%.2e")
            k3  = st.number_input("k3 (decaimiento PFA) [1/s]", value=1.0e-4, format="%.2e")
        with col3:
            k4  = st.number_input("k4 (decaimiento H₂O₂) [1/s]", value=2.0e-5, format="%.2e")
            k5  = st.number_input("k5 (hidrolisis epóxido) [L/mol/s]", value=5.0e-5, format="%.2e")
            alpha = st.number_input("α ácido (factor catalítico)", value=1.0, step=0.1)

    # --- Panel: Transferencia de masa (opcional) ---
    st.subheader("Transferencia de masa (opcional)")
    usar_TM = st.checkbox("Activar transferencia de masa entre fases (aq ↔ org)")
    if usar_TM:
        colTM1, colTM2, colTM3 = st.columns(3)
        with colTM1:
            kla_PFA = st.number_input("kLa_PFA [1/s] (aq→org)", value=5e-3, format="%.2e")
            kla_H2O2 = st.number_input("kLa_H2O2 [1/s] (aq→org)", value=1e-3, format="%.2e")
        with colTM2:
            Kp_PFA = st.number_input("Partición PFA K_oq (=C_org/C_aq)", value=5.0, step=0.5)
            Kp_H2O2 = st.number_input("Partición H₂O₂ K_oq", value=0.05, step=0.01)
        with colTM3:
            frac_aq = st.slider("Fracción de volumen acuoso del total", min_value=0.05, max_value=0.6, value=0.25, step=0.05)
    else:
        kla_PFA = kla_H2O2 = 0.0
        Kp_PFA = 5.0
        Kp_H2O2 = 0.05
        frac_aq = 0.25

    # --- Cálculo de moles iniciales ---
    MW_H2O2=34.0147; MW_HCOOH=46.0254; MW_H2O=18.0153
    g_H2O2 = 0.30 * V_H2O2                # p/v: 30 g por 100 mL
    mol_H2O2_0 = g_H2O2 / MW_H2O2
    g_HCOOH = 0.85 * (V_HCOOH * rho_HCOOH)
    mol_HCOOH_0 = g_HCOOH / MW_HCOOH
    g_H2O = (V_H2O * 1.0) + 0.15*(V_HCOOH*rho_HCOOH) + max(V_H2O2*rho_H2O2 - g_H2O2, 0.0)
    mol_H2O_0 = g_H2O / MW_H2O

    V_total_L = (V_soy + V_H2SO4 + V_HCOOH + V_H2O + V_H2O2)/1000.0
    V_aq = frac_aq * V_total_L
    V_org = V_total_L - V_aq

    # Distribución inicial simple: H₂O₂ y HCOOH comienzan en fase acuosa; PFA=0
    Ca_H2O2_0 = mol_H2O2_0 / V_aq
    Ca_HCOOH_0= mol_HCOOH_0 / V_aq
    Ca_PFA_0  = 0.0
    Co_PFA_0  = 0.0
    Co_CdC_0  = mol_CdC / V_org
    Co_EPOX_0 = 0.0
    Co_OPEN_0 = 0.0
    Ca_H2O_0  = mol_H2O_0 / V_aq

    y0 = np.array([Ca_H2O2_0, Ca_HCOOH_0, Ca_PFA_0, Co_PFA_0, Co_CdC_0, Co_EPOX_0, Co_OPEN_0, Ca_H2O_0])

    @dataclass
    class P:
        k1f: float; k1r: float; k2: float; k3: float; k4: float; k5: float; alpha: float
        kla_PFA: float; kla_H2O2: float; Kp_PFA: float; Kp_H2O2: float
        V_aq: float; V_org: float

    par = P(k1f=k1f, k1r=k1r, k2=k2, k3=k3, k4=k4, k5=k5, alpha=alpha,
            kla_PFA=kla_PFA, kla_H2O2=kla_H2O2, Kp_PFA=Kp_PFA, Kp_H2O2=Kp_H2O2,
            V_aq=V_aq, V_org=V_org)

    # ODEs con dos fases (aq y org):
    # y = [Ca_H2O2, Ca_HCOOH, Ca_PFA, Co_PFA, Co_CdC, Co_EPOX, Co_OPEN, Ca_H2O]
    def rhs(t, y):
        Ca_H2O2, Ca_HCOOH, Ca_PFA, Co_PFA, Co_CdC, Co_EPOX, Co_OPEN, Ca_H2O = y

        # Reacciones en agua: formación/retroceso/decadencia de PFA, pérdida de H2O2
        r1f = par.k1f * Ca_HCOOH * Ca_H2O2 * par.alpha
        r1r = par.k1r * Ca_PFA
        r3  = par.k3  * Ca_PFA
        r4  = par.k4  * Ca_H2O2

        # Epoxidación en orgánico
        r2  = par.k2  * Co_PFA * Co_CdC * par.alpha
        r5  = par.k5  * Co_EPOX * (Ca_H2O) * par.alpha  # hidrolisis por agua “que migra” (aprox.)

        # Transferencia de masa (aq -> org): estilo kLa*(C_aq - C_org/Kp)
        TM_PFA  = par.kla_PFA  * (Ca_PFA  - Co_PFA/par.Kp_PFA)
        TM_H2O2 = par.kla_H2O2 * (Ca_H2O2 - 0.0/par.Kp_H2O2)  # H2O2 ~ no-accumulado en org (≈0)

        dCa_H2O2 = -r1f + r1r - r4 - TM_H2O2
        dCa_HCOOH= -r1f + r1r + r3
        dCa_PFA  =  r1f - r1r - r3 - TM_PFA
        dCo_PFA  =  TM_PFA - r2
        dCo_CdC  = -r2
        dCo_EPOX =  r2 - r5
        dCo_OPEN =  r5
        dCa_H2O  =  r1r + r4  # agua formada en aq

        return [dCa_H2O2, dCa_HCOOH, dCa_PFA, dCo_PFA, dCo_CdC, dCo_EPOX, dCo_OPEN, dCa_H2O]

    st.subheader("Simulación")
    colt1, colt2 = st.columns(2)
    with colt1:
        t_horas = st.number_input("Tiempo total [h]", value=4.0, step=0.5)
    with colt2:
        n_puntos = st.number_input("Puntos de muestreo", value=400, step=50, min_value=50)

    if st.button("▶️ Ejecutar simulación"):
        t_end = t_horas*3600
        t_eval = np.linspace(0, t_end, int(n_puntos))
        sol = solve_ivp(lambda t,y: rhs(t,y), [0, t_end], y0, t_eval=t_eval, method="LSODA", rtol=1e-7, atol=1e-9)

        # Conversión a moles de lote para lectura rápida
        def aq_moles(c): return c*par.V_aq
        def org_moles(c): return c*par.V_org

        # --- Plots (una figura por gráfico) ---
        plt.figure()
        plt.plot(sol.t/3600, aq_moles(sol.y[0]), label="H₂O₂ (aq)")
        plt.plot(sol.t/3600, aq_moles(sol.y[2]), label="PFA (aq)")
        plt.xlabel("Tiempo [h]"); plt.ylabel("moles (lote)")
        plt.title("Oxidantes (fase acuosa)")
        plt.legend(); st.pyplot(plt.gcf())

        plt.figure()
        plt.plot(sol.t/3600, org_moles(sol.y[3]), label="PFA (org)")
        plt.plot(sol.t/3600, org_moles(sol.y[4]), label="C=C (org)")
        plt.plot(sol.t/3600, org_moles(sol.y[5]), label="Epóxido (org)")
        plt.plot(sol.t/3600, org_moles(sol.y[6]), label="Apertura (org)")
        plt.xlabel("Tiempo [h]"); plt.ylabel("moles (lote)")
        plt.title("Orgánico: PFA, C=C, Epóxido y apertura")
        plt.legend(); st.pyplot(plt.gcf())

        # Resumen numérico
        st.markdown("### Resumen final (moles en el lote)")
        st.write({
            "H2O2_final_mol": float(aq_moles(sol.y[0,-1])),
            "PFA_aq_final_mol": float(aq_moles(sol.y[2,-1])),
            "PFA_org_final_mol": float(org_moles(sol.y[3,-1])),
            "Epox_final_mol": float(org_moles(sol.y[5,-1])),
            "Open_final_mol": float(org_moles(sol.y[6,-1]))
        })

    # Sector flotante (tu helper)
    mostrar_sector_flotante(db, key_suffix="modelo")
