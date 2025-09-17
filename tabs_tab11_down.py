# tabs_tab11_down.py
import streamlit as st
import pandas as pd
import hashlib, json, re
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global

# ===== Config =====
ETAPAS = [1, 2, 3, 4, 5, 6]  # ampliado a 6
CAMPOS_ETAPA = ["Sal", "[]", "V (mL)", "T", "t ag (h)", "t dec (h)", "V dec (mL)"]

def _new_cols_etapa(n:int):
    return [f"{n}_{c}" for c in CAMPOS_ETAPA]

BASE_COLS = (
    ["Síntesis"] +             # ID
    ["$"] +                    # NUEVA columna calculada
    ["VOL ACUO (mL)"] +
    [c for e in ETAPAS for c in _new_cols_etapa(e)] +
    ["Observaciones"]
)

# Mapa de migración antiguo -> nuevo
OLD2NEW = lambda n: {
    f"{n}_Sal":             f"E{n} TIPO DE SAL",
    f"{n}_[]":              f"E{n} CONCENTRACION",
    f"{n}_V (mL)":          f"E{n} VOLUMEN (mL)",
    f"{n}_T":               f"E{n} TEMP (°C)",
    f"{n}_t ag (h)":        f"E{n} t AGIT (h)",
    f"{n}_t dec (h)":       f"E{n} t DECAN (h)",
    f"{n}_V dec (mL)":      f"E{n} FASE ACUO RET (mL)",
}

# ===== Utils =====
def _df_vacio():
    return pd.DataFrame(columns=BASE_COLS)

def _hash_rows(rows):
    s = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def _migrar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea nuevas columnas y, si existen, copia datos desde las viejas."""
    for n in ETAPAS:
        mapping = OLD2NEW(n)
        for new_col, old_col in mapping.items():
            if new_col not in df.columns:
                if old_col in df.columns:
                    df[new_col] = df[old_col]
                else:
                    df[new_col] = ""
    # eliminar columnas viejas si quedaron
    cols_drop = []
    for n in ETAPAS:
        cols_drop += list(OLD2NEW(n).values())
    df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")
    return df

def _parse_float(x):
    try:
        if isinstance(x, (int, float)): return float(x)
        if x is None: return None
        s = str(x).strip().replace(",", ".")
        # tomar primer número que aparezca
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        return float(m.group(0)) if m else None
    except Exception:
        return None

def _kg_soluto(conc_str, vol_ml):
    """
    Devuelve kg de soluto a partir de:
      - conc en % (p.ej. 5 o '5%')  => masa soluto = fracción * masa solución; 1 L ~= 1 kg
      - conc en g/L (p.ej. '10 g/L') => masa soluto = (g/L)*L / 1000
    Cualquier otro formato => 0.
    """
    if conc_str is None or conc_str == "": return 0.0
    v_l = (_parse_float(vol_ml) or 0.0) / 1000.0  # mL -> L
    s = str(conc_str).strip().lower().replace(",", ".")
    if "g/l" in s:
        gpl = _parse_float(s) or 0.0
        return (gpl * v_l) / 1000.0
    # asume % si hay '%' o si es número pelado
    pct = _parse_float(s)
    if pct is not None:
        frac = pct / 100.0
        return frac * v_l  # ~ kg (porque 1 L solución ~ 1 kg)
    return 0.0

def _calcular_coste_fila(row: pd.Series, precio_kg: dict) -> float:
    total = 0.0
    for n in ETAPAS:
        agente = str(row.get(f"{n}_Sal", "")).strip()
        if not agente:
            continue
        precio = _parse_float(precio_kg.get(agente))
        if not precio:  # sin precio conocido => 0
            continue
        conc = row.get(f"{n}_[]", "")
        vol  = row.get(f"{n}_V (mL)", "")
        kg = _kg_soluto(conc, vol)
        total += precio * kg
    return round(total, 2)

# ===== UI =====
def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Down"

    # --- Cargar Firestore ---
    datos = cargar_sintesis_global(db) or {}
    # principal
    raw = datos.get("down_tabla") or []
    df_in = pd.DataFrame(raw) if raw else _df_vacio()

    # costos secundarios (persisten en 'down_costos_agentes')
    raw_costos = (datos.get("down_costos_agentes") or [])
    if raw_costos:
        df_costos = pd.DataFrame(raw_costos)
    else:
        df_costos = pd.DataFrame({
            "Agente": ["Na2SO3", "NaHCO3", "NaOH", "H2O"],
            "Costo (por kg)": ["", "", "", ""],
        })
    # asegurar columnas costos
    for c in ["Agente", "Costo (por kg)"]:
        if c not in df_costos.columns:
            df_costos[c] = ""
    df_costos = df_costos.fillna("").astype(str)
    # mapa de precios
    precio_kg = {str(a).strip(): v for a, v in zip(df_costos["Agente"], df_costos["Costo (por kg)"])}

    # asegurar columnas mínimas en principal
    for c in BASE_COLS:
        if c not in df_in.columns:
            df_in[c] = ""
    # migración (no afecta 5-6 si no existen)
    df_in = _migrar_columnas(df_in)

    # calcular columna '$' (solo lectura)
    # primero garantizar orden y tipos
    # ojo: '$' se recalcula siempre y NO se persiste
    df_in = df_in[[c for c in BASE_COLS if c != "$"]]  # quitar si existía
    # insert '$' tras 'Síntesis'
    cols = df_in.columns.tolist()
    if "Síntesis" in cols:
        idx = cols.index("Síntesis") + 1
        df_in.insert(idx, "$", "")
    else:
        df_in.insert(0, "$", "")
    # calcular
    costos = []
    for _, r in df_in.iterrows():
        costos.append(_calcular_coste_fila(r, precio_kg))
    df_in["$"] = costos

    df_in = df_in.fillna("").astype(str)
    # reordenar explícitamente al esquema base
    df_in = df_in[[c for c in BASE_COLS]]

    # ---- Column config (igual que antes, '$' deshabilitada) ----
    colcfg = {c: st.column_config.TextColumn(label=c, width="small") for c in BASE_COLS}
    colcfg["Síntesis"] = st.column_config.TextColumn(label="ID", width="small")
    colcfg["$"] = st.column_config.TextColumn(label="$", help="Costo total calculado", width="small")
    colcfg["VOL ACUO (mL)"] = st.column_config.TextColumn(label="Vaq (mL)", help="Volumen fase acuosa", width="small")
    colcfg["Observaciones"] = st.column_config.TextColumn(label="Obs", width="large")
    for n in (1, 2, 3, 4, 5, 6):
        colcfg[f"{n}_Sal"]        = st.column_config.TextColumn(label=f"{n}_Ag",   help=f"Etapa {n}: Sal", width="small")
        colcfg[f"{n}_[]"]         = st.column_config.TextColumn(label=f"{n}_[]",   help=f"Etapa {n}: concentración", width="small")
        colcfg[f"{n}_V (mL)"]     = st.column_config.TextColumn(label=f"{n}_V",    help=f"Etapa {n}: Volumen (mL)", width="small")
        colcfg[f"{n}_T"]          = st.column_config.TextColumn(label=f"{n}_T",    help=f"Etapa {n}: Temperatura (°C)", width="small")
        colcfg[f"{n}_t ag (h)"]   = st.column_config.TextColumn(label=f"{n}_tAg",  help=f"Etapa {n}: tiempo de agitación (h)", width="small")
        colcfg[f"{n}_t dec (h)"]  = st.column_config.TextColumn(label=f"{n}_tDec", help=f"Etapa {n}: tiempo de decantación (h)", width="small")
        colcfg[f"{n}_V dec (mL)"] = st.column_config.TextColumn(label=f"{n}_Vdec", help=f"Etapa {n}: volumen decantado (mL)", width="small")

    # ---- Estilo encabezados (igual) ----
    st.markdown("""
    <style>
    div[data-testid="stDataEditorGrid"] thead th,
    div[data-testid="stDataFrame"] thead th{
      padding: 2px 4px !important; white-space: nowrap !important;
      overflow: hidden !important; text-overflow: ellipsis !important;
    }
    div[data-testid="stDataEditorGrid"] thead svg,
    div[data-testid="stDataFrame"] thead svg{ display: none !important; }
    </style>
    """, unsafe_allow_html=True)

    # ---- Hash base (sin '$' para evitar guardados en bucle) ----
    rows_base = df_in.drop(columns=["$"], errors="ignore").fillna("").to_dict("records")
    if "down_hash" not in st.session_state:
        st.session_state["down_hash"] = _hash_rows(rows_base)
    if "down_costos_hash" not in st.session_state:
        st.session_state["down_costos_hash"] = _hash_rows(df_costos.to_dict("records"))

    # ---- Editores ----
    df_edit = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        column_config=colcfg,
        hide_index=True,
        key="down_editor_native",
        disabled=["$"],  # '$' solo lectura
    )

    # tabla secundaria (simple, debajo)
    st.subheader("Costos por kg de agentes")
    df_costos_edit = st.data_editor(
        df_costos,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Agente": st.column_config.TextColumn(label="Agente", width="medium"),
            "Costo (por kg)": st.column_config.TextColumn(label="Costo (por kg)", width="small"),
        },
        key="down_costos_editor",
    )

    # ---- Guardado (principal sin '$' + costos) ----
    rows = df_edit.fillna("").to_dict("records")
    # quitar '$' antes de persistir
    rows_save = [{k: v for k, v in r.items() if k != "$"} for r in rows]
    h = _hash_rows(rows_save)

    rows_costos = df_costos_edit.fillna("").to_dict("records")
    h_costos = _hash_rows(rows_costos)

    changed = False
    payload = {**datos}
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
