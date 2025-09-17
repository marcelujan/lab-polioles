# tabs_tab11_down.py
import streamlit as st
import pandas as pd
import hashlib, json, re
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global

# ===== Config =====
ETAPAS = [1, 2, 3, 4, 5, 6]  # hasta 6 pasos
CAMPOS_ETAPA = ["Sal", "[]", "V (mL)", "T", "t ag (h)", "t dec (h)", "V dec (mL)"]

def _new_cols_etapa(n:int):
    return [f"{n}_{c}" for c in CAMPOS_ETAPA]

# '$' va entre ID y Vaq (mL)
BASE_COLS = (
    ["Síntesis", "$", "VOL ACUO (mL)"] +
    [c for e in ETAPAS for c in _new_cols_etapa(e)] +
    ["Observaciones"]
)

# migración desde esquema viejo
OLD2NEW = lambda n: {
    f"{n}_Sal":             f"E{n} TIPO DE SAL",
    f"{n}_[]":              f"E{n} CONCENTRACION",
    f"{n}_V (mL)":          f"E{n} VOLUMEN (mL)",
    f"{n}_T":               f"E{n} TEMP (°C)",
    f"{n}_t ag (h)":        f"E{n} t AGIT (h)",
    f"{n}_t dec (h)":       f"E{n} t DECAN (h)",
    f"{n}_V dec (mL)":      f"E{n} FASE ACUO RET (mL)",
}

# tabla secundaria
COSTOS_COLS = ["Agente", "Costo (por kg)"]  # enteros

# ===== Utils =====
def _df_vacio():
    return pd.DataFrame(columns=BASE_COLS)

def _hash_rows(rows):
    s = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _migrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    for n in ETAPAS:
        mapping = OLD2NEW(n)
        for new_col, old_col in mapping.items():
            if new_col not in df.columns:
                df[new_col] = df[old_col] if old_col in df.columns else ""
    cols_drop = []
    for n in ETAPAS:
        cols_drop += list(OLD2NEW(n).values())
    return df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")

def _parse_float(x):
    try:
        if isinstance(x, (int, float)): return float(x)
        s = str(x).strip().replace(",", ".")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else None
    except Exception:
        return None

def _kg_soluto_pct_pp(conc_pct, vol_ml):
    """conc en % p/p (acepta '5' o '5%'), ρ≈1 kg/L."""
    if conc_pct is None or conc_pct == "": return 0.0
    pct = _parse_float(conc_pct)
    if pct is None: return 0.0
    v_l = (_parse_float(vol_ml) or 0.0) / 1000.0  # mL -> L
    return (pct/100.0) * v_l  # kg de soluto

def _calcular_coste_fila(row: pd.Series, precio_kg: dict) -> int:
    total = 0.0
    for n in ETAPAS:
        agente = str(row.get(f"{n}_Sal", "")).strip()
        if not agente: continue
        precio = _parse_float(precio_kg.get(agente))
        if not precio: continue
        conc = row.get(f"{n}_[]", "")
        vol  = row.get(f"{n}_V (mL)", "")
        kg = _kg_soluto_pct_pp(conc, vol)
        total += precio * kg
    return int(round(total))  # sin decimales

# ===== UI =====
def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Down"

    datos = cargar_sintesis_global(db) or {}

    # ---------- principal ----------
    raw = datos.get("down_tabla") or []
    df_in = pd.DataFrame(raw) if raw else _df_vacio()

    # asegurar mínimas + migración
    for c in BASE_COLS:
        if c not in df_in.columns: df_in[c] = ""
    df_in = _migrar_columnas(df_in)

    # ---------- costos secundarios (persistentes) ----------
    raw_costos = (datos.get("down_costos_agentes") or [])
    if raw_costos:
        df_costos = pd.DataFrame(raw_costos)
    else:
        df_costos = pd.DataFrame({
            "Agente": ["Na2SO3", "NaHCO3", "NaOH", "H2O"],
            "Costo (por kg)": ["", "", "", ""],
        })
    # Normalizar columnas/orden (por si alguna vez quedaron invertidas)
    for c in COSTOS_COLS:
        if c not in df_costos.columns:
            df_costos[c] = ""
    df_costos = df_costos[COSTOS_COLS].fillna("").astype(str)

    # Diccionario de precios (robusto al orden)
    precio_kg = {}
    for _, r in df_costos.iterrows():
        agente = str(r.get("Agente", "")).strip()
        precio = _parse_float(r.get("Costo (por kg)", ""))
        if agente and precio is not None:
            precio_kg[agente] = precio

    # ---------- calcular '$' y ordenar columnas ----------
    # quitar '$' si existía y volver a insertarla tras 'Síntesis'
    df_in = df_in[[c for c in df_in.columns if c != "$"]]
    idx = (df_in.columns.tolist().index("Síntesis")+1) if "Síntesis" in df_in.columns else 0
    df_in.insert(idx, "$", 0)
    df_in["$"] = [ _calcular_coste_fila(r, precio_kg) for _, r in df_in.iterrows() ]
    df_in = df_in.fillna("").astype(str)
    df_in = df_in[[c for c in BASE_COLS]]

    # ---------- estilos ----------
    st.markdown("""
    <style>
    div[data-testid="stDataEditorGrid"] thead th,
    div[data-testid="stDataFrame"] thead th{
      padding:2px 4px!important; white-space:nowrap!important; overflow:hidden!important; text-overflow:ellipsis!important;
    }
    div[data-testid="stDataEditorGrid"] thead svg,
    div[data-testid="stDataFrame"] thead svg{ display:none!important; }
    </style>
    """, unsafe_allow_html=True)

    # ---------- column config ----------
    colcfg = {c: st.column_config.TextColumn(label=c, width="small") for c in BASE_COLS}
    colcfg["Síntesis"] = st.column_config.TextColumn(label="ID", width="small")
    colcfg["$"] = st.column_config.NumberColumn(label="$", width="small", step=1, format="%.0f")
    colcfg["VOL ACUO (mL)"] = st.column_config.TextColumn(label="Vaq (mL)", help="Volumen fase acuosa", width="small")
    colcfg["Observaciones"] = st.column_config.TextColumn(label="Obs", width="large")
    for n in ETAPAS:
        colcfg[f"{n}_Sal"]        = st.column_config.TextColumn(label=f"{n}_Ag",   width="small")
        colcfg[f"{n}_[]"]         = st.column_config.TextColumn(label=f"{n}_[]",   width="small")
        colcfg[f"{n}_V (mL)"]     = st.column_config.TextColumn(label=f"{n}_V",    width="small")
        colcfg[f"{n}_T"]          = st.column_config.TextColumn(label=f"{n}_T",    width="small")
        colcfg[f"{n}_t ag (h)"]   = st.column_config.TextColumn(label=f"{n}_tAg",  width="small")
        colcfg[f"{n}_t dec (h)"]  = st.column_config.TextColumn(label=f"{n}_tDec", width="small")
        colcfg[f"{n}_V dec (mL)"] = st.column_config.TextColumn(label=f"{n}_Vdec", width="small")

    # ---------- hashes (sin '$' para evitar bucles) ----------
    rows_base = df_in.drop(columns=["$"], errors="ignore").fillna("").to_dict("records")
    if "down_hash" not in st.session_state:
        st.session_state["down_hash"] = _hash_rows(rows_base)
    if "down_costos_hash" not in st.session_state:
        st.session_state["down_costos_hash"] = _hash_rows(df_costos.to_dict("records"))

    # ---------- editor principal ----------
    df_edit = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        column_config=colcfg,
        hide_index=True,
        key="down_editor_native",
        disabled=["$"],  # solo lectura
    )

    # ---------- editor costos ----------
    st.subheader("Costos por kg de agentes")
    df_costos_edit = st.data_editor(
        df_costos,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Agente": st.column_config.TextColumn(label="Agente", width="medium"),
            "Costo (por kg)": st.column_config.NumberColumn(label="Costo (por kg)", step=1, format="%.0f", width="small"),
        },
        key="down_costos_editor",
    )

    # ---------- guardado (principal sin '$' + costos) ----------
    rows = df_edit.fillna("").to_dict("records")
    rows_save = [{k:v for k,v in r.items() if k != "$"} for r in rows]
    h = _hash_rows(rows_save)

    rows_costos = df_costos_edit.fillna("").to_dict("records")
    h_costos = _hash_rows(rows_costos)

    changed, payload = False, {**datos}
    if h != st.session_state["down_hash"]:
        payload["down_tabla"] = rows_save
        st.session_state["down_hash"] = h
        changed = True
    if h_costos != st.session_state["down_costos_hash"]:
        payload["down_costos_agentes"] = rows_costos
        st.session_state["down_costos_hash"] = h_costos
        changed = True

    if changed:
        guardar_sintesis_global(db, payload)
        st.toast("Guardado", icon="✅")
