# tabs_tab10_modelo.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.integrate import solve_ivp

# ──────────────────────────────────────────────────────────────────────────────
#  Pestaña: Modelo cinético – Mi PoliOL (ecuaciones explícitas)
# ──────────────────────────────────────────────────────────────────────────────

def render_tab10(db=None, mostrar_sector_flotante=lambda *a, **k: None):
    st.title("Modelo cinético – Mi PoliOL (ecuaciones explícitas)")
    st.session_state["current_tab"] = "Modelo cinético"

    st.markdown("""
### Esquema de reacción (homog. mínima)
**(R1)** Formación de perácido (perfórmico)  
\\[
\\mathrm{HCOOH + H_2O_2 \\xrightleftharpoons[k_{1r}]{k_{1f}} PFA + H_2O}
\\]

**(R2)** Epoxidación (fase orgánica)  
\\[
\\mathrm{PFA + C{=}C \\xrightarrow{k_2} Ep\\ (epóxido) + HCOOH}
\\]

**(R3)** Decaimiento del perácido  
\\[
\\mathrm{PFA \\xrightarrow{k_3} HCOOH}
\\]

**(R4)** Decaimiento del peróxido  
\\[
\\mathrm{H_2O_2 \\xrightarrow{k_4} H_2O}
\\]

**(R5)** Apertura del epóxido por agua  
\\[
\\mathrm{Ep + H_2O \\xrightarrow{k_5} Open}
\\]

**Suposiciones mínimas:** mezcla líquida isoterma, volumen constante, catálisis ácida lumped (\\(\\alpha\\)), sin pérdida de masa por fases gaseosas.  
**Opción TM (transferencia de masa):** dos fases (acuosa/ orgánica) con **dos-películas**:  
\\[
\\mathrm{N_i = k_L a\\,(C_{i,aq}-C_{i,org}/K_{oq})}
\\]
    """)

    # ── Entradas de composición (Mi PoliOL)
    st.subheader("Composición inicial (volúmenes por lote)")
    c1, c2, c3 = st.columns(3)
    with c1:
        V_soy = st.number_input("Aceite de soja crudo [mL]",  value=400.00, step=1.0)
        V_H2SO4 = st.number_input("Ácido sulfúrico 98% [mL]", value=3.64,  step=0.1)
        V_H2O = st.number_input("Agua destilada [mL]",        value=32.73, step=0.1)
    with c2:
        V_HCOOH = st.number_input("Ácido fórmico 85% [mL]",   value=80.00, step=0.1)
        V_H2O2  = st.number_input("Peróxido 30% **p/v** [mL]", value=204.36, step=0.1,
                                   help="30 g H2O2 / 100 mL de solución (p/v)")
        moles_CdC = st.number_input("Moles de C=C (lote)", value=2.00, step=0.1,
                                    help="Ajustar según IY/oxirano objetivo")
    with c3:
        st.markdown("**Densidades (si querés precisión, editá):**")
        rho_soy = st.number_input("ρ aceite [g/mL]",     value=0.92,  step=0.01)
        rho_HCOOH=st.number_input("ρ HCOOH 85% [g/mL]",  value=1.215, step=0.005)
        rho_H2O2 =st.number_input("ρ H₂O₂ 30% p/v [g/mL]", value=1.00, step=0.01)

    # Cálculo de moles iniciales (explícito)
    st.markdown("""
**Cálculo de moles iniciales**  
Para **H₂O₂ 30% p/v**: \\(m_{H_2O_2}=0.30\\,V_{H_2O_2}\\) (g) y \\(n=m/MW\\).  
Para **HCOOH 85% p/p**: \\(m_{HCOOH}=0.85\\,\\rho_{85}\\,V_{85}\\).
    """)

    MW_H2O2=34.0147; MW_HCOOH=46.0254; MW_H2O=18.0153
    g_H2O2 = 0.30 * V_H2O2
    n_H2O2 = g_H2O2 / MW_H2O2
    g_HCOOH = 0.85 * rho_HCOOH * V_HCOOH
    n_HCOOH = g_HCOOH / MW_HCOOH
    g_H2O_ini = (V_H2O*1.0) + 0.15*(rho_HCOOH*V_HCOOH) + max(rho_H2O2*V_H2O2 - g_H2O2, 0.0)
    n_H2O = g_H2O_ini / MW_H2O

    V_total_L = (V_soy + V_H2SO4 + V_H2O + V_HCOOH + V_H2O2)/1000.0

    # ── Cinética
    st.subheader("Constantes cinéticas y factor ácido")
    k1f = st.number_input("k1f [L/mol/s] (R1→)", value=2.0e-2, format="%.2e")
    k1r = st.number_input("k1r [1/s] (R1←)",     value=1.0e-3, format="%.2e")
    k2  = st.number_input("k2  [L/mol/s] (R2)",  value=1.0e-2, format="%.2e")
    k3  = st.number_input("k3  [1/s] (R3)",      value=1.0e-4, format="%.2e")
    k4  = st.number_input("k4  [1/s] (R4)",      value=2.0e-5, format="%.2e")
    k5  = st.number_input("k5  [L/mol/s] (R5)",  value=5.0e-5, format="%.2e")
    alpha = st.number_input("α (catal. ácida lumped)", value=1.0, step=0.1)

    st.markdown(r"""
**Ecuaciones diferenciales (formulación 1-fase, concentraciones \\(C\\) en mol·L⁻¹):**

\\[
\\begin{aligned}
\\frac{dC_{H_2O_2}}{dt} &= -k_{1f}\\,C_{HCOOH}C_{H_2O_2}\\,\\alpha + k_{1r}C_{PFA} - k_4 C_{H_2O_2} \\\\
\\frac{dC_{HCOOH}}{dt} &= -k_{1f}\\,C_{HCOOH}C_{H_2O_2}\\,\\alpha + k_{1r}C_{PFA} + k_2 C_{PFA}C_{C{=}C}\\,\\alpha + k_3 C_{PFA} \\\\
\\frac{dC_{PFA}}{dt} &= k_{1f}C_{HCOOH}C_{H_2O_2}\\,\\alpha - k_{1r}C_{PFA} - k_2 C_{PFA}C_{C{=}C}\\,\\alpha - k_3 C_{PFA} \\\\
\\frac{dC_{C{=}C}}{dt} &= -k_2 C_{PFA}C_{C{=}C}\\,\\alpha \\\\
\\frac{dC_{Ep}}{dt} &= k_2 C_{PFA}C_{C{=}C}\\,\\alpha - k_5 C_{Ep}C_{H_2O}\\,\\alpha \\\\
\\frac{dC_{Open}}{dt} &= k_5 C_{Ep}C_{H_2O}\\,\\alpha
\\end{aligned}
\\]

**Balance de agua (simplificado):** \\(\\frac{dC_{H_2O}}{dt} = k_{1r}C_{PFA} + k_4 C_{H_2O_2}\\).
    """)

    # Toggle de transferencia de masa
    usar_TM = st.checkbox("Usar **dos fases** con transferencia de masa (dos-películas)")
    if usar_TM:
        st.markdown(r"""
**Reparto de volumen:** \\(V=V_{aq}+V_{org}\\).  
**Flujo de TM:** \\(\\dot n_i = k_L a\\,(C_{i,aq}-C_{i,org}/K_{oq})\\).
        """)
        coltm = st.columns(3)
        with coltm[0]:
            frac_aq = st.slider("Fracción acuosa Vaq/V", 0.05, 0.60, 0.25, 0.05)
        with coltm[1]:
            kla_PFA = st.number_input("kLa_PFA [1/s]", value=5e-3, format="%.2e")
            Kp_PFA  = st.number_input("K_oq PFA (=Corg/Caq)", value=5.0, step=0.5)
        with coltm[2]:
            kla_H2O2 = st.number_input("kLa_H2O2 [1/s]", value=1e-3, format="%.2e")
            Kp_H2O2  = st.number_input("K_oq H2O2", value=0.05, step=0.01)
    else:
        frac_aq=0.25; kla_PFA=kla_H2O2=0.0; Kp_PFA=5.0; Kp_H2O2=0.05

    # Estados iniciales
    V_aq = frac_aq*V_total_L; V_org = V_total_L - V_aq
    if usar_TM:
        # distribución: H2O2 y HCOOH en aq; C=C en org; PFA=0
        y0 = np.array([
            n_H2O2/V_aq,           # Ca_H2O2
            n_HCOOH/V_aq,          # Ca_HCOOH
            0.0,                   # Ca_PFA
            0.0,                   # Co_PFA
            moles_CdC/V_org,       # Co_CdC
            0.0, 0.0,              # Co_Ep, Co_Open
            n_H2O/V_aq             # Ca_H2O
        ])
    else:
        # 1 fase
        y0 = np.array([
            n_H2O2/V_total_L,
            n_HCOOH/V_total_L,
            0.0,
            moles_CdC/V_total_L,
            0.0, 0.0,
            n_H2O/V_total_L
        ])

    @dataclass
    class P:
        k1f: float; k1r: float; k2: float; k3: float; k4: float; k5: float; alpha: float
        Vaq: float; Vorg: float; kla_PFA: float; kla_H2O2: float; Kp_PFA: float; Kp_H2O2: float
        usar_TM: bool

    par = P(k1f, k1r, k2, k3, k4, k5, alpha, V_aq, V_org, kla_PFA, kla_H2O2, Kp_PFA, Kp_H2O2, usar_TM)

    # RHS explícito (muestra fórmulas en texto en la UI)
    with st.expander("Ecuaciones en forma de código (para auditoría)"):
        st.code("""
dH2O2 = -k1f*HCOOH*H2O2*alpha + k1r*PFA - k4*H2O2
dHCOOH= -k1f*HCOOH*H2O2*alpha + k1r*PFA + k2*PFA*CdC*alpha + k3*PFA
dPFA  =  k1f*HCOOH*H2O2*alpha - k1r*PFA - k2*PFA*CdC*alpha - k3*PFA
dCdC  = -k2*PFA*CdC*alpha
dEp   =  k2*PFA*CdC*alpha - k5*Ep*H2O*alpha
dOpen =  k5*Ep*H2O*alpha
dH2O  =  k1r*PFA + k4*H2O2
# Con TM: + kLa*(Caq - Corg/Koq) en PFA y H2O2, y balance en ambas fases
        """, language="python")

    # Sistema ODE
    def rhs_1phase(t, y, p: P):
        H2O2, HCOOH, PFA, CdC, Ep, Open, H2O = y
        r1f = p.k1f*HCOOH*H2O2*p.alpha; r1r=p.k1r*PFA
        r2  = p.k2*PFA*CdC*p.alpha;      r3 = p.k3*PFA
        r4  = p.k4*H2O2;                 r5 = p.k5*Ep*H2O*p.alpha
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
        # Aq
        r1f = p.k1f*Ca_HCOOH*Ca_H2O2*p.alpha; r1r=p.k1r*Ca_PFA; r3=p.k3*Ca_PFA; r4=p.k4*Ca_H2O2
        # Org
        r2  = p.k2*Co_PFA*Co_CdC*p.alpha;     r5=p.k5*Co_Ep*Ca_H2O*p.alpha
        # TM
        TM_PFA  = p.kla_PFA*(Ca_PFA - Co_PFA/p.Kp_PFA)
        TM_H2O2 = p.kla_H2O2*(Ca_H2O2 - 0.0/p.Kp_H2O2)

        return [
            -r1f + r1r - r4 - TM_H2O2,         # dCa_H2O2
            -r1f + r1r + r3,                   # dCa_HCOOH
             r1f - r1r - r3 - TM_PFA,          # dCa_PFA
             TM_PFA - r2,                      # dCo_PFA
            -r2,                               # dCo_CdC
             r2 - r5,                          # dCo_Ep
             r5,                               # dCo_Open
             r1r + r4                          # dCa_H2O
        ]

    # Tiempo de simulación
    st.subheader("Simulación")
    cols = st.columns(2)
    with cols[0]:
        t_h = st.number_input("Tiempo total [h]", 4.0, step=0.5)
    with cols[1]:
        npts = st.number_input("Puntos", 400, step=50, min_value=100)

    if st.button("▶ Ejecutar"):
        t_end = t_h*3600; t_eval = np.linspace(0, t_end, int(npts))
        if usar_TM:
            sol = solve_ivp(lambda t,Y: rhs_2phase(t,Y,par), [0,t_end], y0, t_eval=t_eval,
                            method="LSODA", rtol=1e-7, atol=1e-9)
            # Conversión a moles de lote para lectura
            def aqmol(c): return c*par.Vaq
            def orgmol(c):return c*par.Vorg

            # Gráfico oxidantes (aq)
            plt.figure(); plt.plot(sol.t/3600, aqmol(sol.y[0]), label="H₂O₂ (aq)")
            plt.plot(sol.t/3600, aqmol(sol.y[2]), label="PFA (aq)")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles (lote)"); plt.title("Oxidantes – fase acuosa")
            plt.legend(); st.pyplot(plt.gcf())

            # Orgánico
            plt.figure()
            plt.plot(sol.t/3600, orgmol(sol.y[3]), label="PFA (org)")
            plt.plot(sol.t/3600, orgmol(sol.y[4]), label="C=C (org)")
            plt.plot(sol.t/3600, orgmol(sol.y[5]), label="Epóxido (org)")
            plt.plot(sol.t/3600, orgmol(sol.y[6]), label="Apertura (org)")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles (lote)"); plt.title("Orgánico – PFA, C=C, Ep, Open")
            plt.legend(); st.pyplot(plt.gcf())

            st.markdown("**Resumen final (moles):**")
            st.write({
                "H2O2_aq_final": float(aqmol(sol.y[0,-1])),
                "PFA_aq_final": float(aqmol(sol.y[2,-1])),
                "PFA_org_final": float(orgmol(sol.y[3,-1])),
                "Epox_org_final": float(orgmol(sol.y[5,-1])),
                "Open_org_final": float(orgmol(sol.y[6,-1]))
            })

        else:
            sol = solve_ivp(lambda t,Y: rhs_1phase(t,Y,par), [0,t_end], y0, t_eval=t_eval,
                            method="LSODA", rtol=1e-7, atol=1e-9)

            def mol(c): return c*V_total_L
            # Oxidantes
            plt.figure()
            plt.plot(sol.t/3600, mol(sol.y[0]), label="H₂O₂")
            plt.plot(sol.t/3600, mol(sol.y[2]), label="PFA")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles (lote)")
            plt.title("Oxidantes (modelo 1-fase)")
            plt.legend(); st.pyplot(plt.gcf())

            # Sustrato/epóxido
            plt.figure()
            plt.plot(sol.t/3600, mol(sol.y[3]), label="C=C")
            plt.plot(sol.t/3600, mol(sol.y[4]), label="Epóxido")
            plt.plot(sol.t/3600, mol(sol.y[5]), label="Apertura")
            plt.xlabel("Tiempo [h]"); plt.ylabel("moles (lote)")
            plt.title("C=C / Epóxido / Apertura")
            plt.legend(); st.pyplot(plt.gcf())

            st.markdown("**Resumen final (moles):**")
            st.write({
                "H2O2_final": float(mol(sol.y[0,-1])),
                "PFA_final": float(mol(sol.y[2,-1])),
                "Epox_final": float(mol(sol.y[4,-1])),
                "Open_final": float(mol(sol.y[5,-1]))
            })

    st.markdown("""
---
### Simplificaciones explícitas
- Volumen constante, isoterma, sin evaporación ni arrastre gaseoso.  
- Catálisis ácida agrupada en \\(\\alpha\\).  
- **1-fase:** todos los reactivos bien mezclados.  
- **2-fases (opcional):** reparto fijo de volúmenes y TM con \\(k_La\\) y \\(K_{oq}\\) constantes.  
- No se modelan impurezas ni efectos de actividad; tasas de R1–R5 son de **ley de potencia**.
    """)

    mostrar_sector_flotante(db, key_suffix="modelo_exp")
