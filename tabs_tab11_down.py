# tabs_tab11_down.py
import streamlit as st
import pandas as pd
import hashlib, json
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global

ETAPAS = [1, 2, 3, 4]

CAMPOS_ETAPA = ["Agente", "V (mL)", "T", "t ag (h)", "t dec (h)", "V dec (mL)"]

def _new_cols_etapa(n:int):
    return [f"{n}_{c}" for c in CAMPOS_ETAPA]

BASE_COLS = (
    ["Síntesis", "VOL ACUO (mL)"] +
    [c for e in ETAPAS for c in _new_cols_etapa(e)] +
    ["Observaciones"]
)

OLD2NEW = lambda n: {
    f"{n}_Agente":          f"E{n} TIPO DE SAL",
    f"{n}_V (mL)":          f"E{n} VOLUMEN (mL)",
    f"{n}_T":               f"E{n} TEMP (°C)",
    f"{n}_t ag (h)":        f"E{n} t AGIT (h)",
    f"{n}_t dec (h)":       f"E{n} t DECAN (h)",
    f"{n}_V dec (mL)":      f"E{n} FASE ACUO RET (mL)",
}

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
    # Opcional: podés conservar las viejas; aquí las eliminamos si existen
    cols_drop = []
    for n in ETAPAS:
        cols_drop += list(OLD2NEW(n).values())
    df = df.drop(columns=[c for c in cols_drop if c in df.columns], errors="ignore")
    return df

def render_tab11(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.session_state["current_tab"] = "Down"

    # --- Cargar desde Firestore ---
    datos = cargar_sintesis_global(db) or {}
    raw = datos.get("down_tabla") or []
    df_in = pd.DataFrame(raw) if raw else _df_vacio()

    # Asegurar columnas mínimas
    for c in BASE_COLS:
        if c not in df_in.columns:
            df_in[c] = ""

    # Migración desde el esquema viejo (si aplica)
    df_in = _migrar_columnas(df_in)

    # Reordenar columnas al esquema nuevo
    df_in = df_in[[c for c in BASE_COLS]]

    colcfg = {c: st.column_config.TextColumn(label=c, width="small") for c in BASE_COLS}

    # Etiquetas compactas opcionales:
    colcfg["Síntesis"] = st.column_config.TextColumn(label="ID", width="small")
    colcfg["VOL ACUO (mL)"] = st.column_config.TextColumn(label="Vaq (mL)", help="Volumen fase acuosa", width="small")
    colcfg["Observaciones"] = st.column_config.TextColumn(label="Obs", width="large")

    for n in (1,2,3,4):
        colcfg[f"{n}_Agente"]     = st.column_config.TextColumn(label=f"{n}_Ag",   help=f"Etapa {n}: Agente", width="small")
        colcfg[f"{n}_V (mL)"]     = st.column_config.TextColumn(label=f"{n}_V",  help=f"Etapa {n}: Volumen (mL)", width="small", step=1)
        colcfg[f"{n}_T"]          = st.column_conffig.TextColumn(label=f"{n}_T",  help=f"Etapa {n}: Temperatura (°C)", width="small", step=1)
        colcfg[f"{n}_t ag (h)"]   = st.column_config.TextColumn(label=f"{n}_tAg",help=f"Etapa {n}: tiempo de agitación (h)", width="small", step=0.1, format="%.2f")
        colcfg[f"{n}_t dec (h)"]  = st.column_config.TextColumn(label=f"{n}_tDec",help=f"Etapa {n}: tiempo de decantación (h)", width="small", step=0.1, format="%.2f")
        colcfg[f"{n}_V dec (mL)"] = st.column_config.TextColumn(label=f"{n}_Vdec",help=f"Etapa {n}: volumen decantado (mL)", width="small", step=1)

    # --- encabezados compactos ---
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

    # hash base de lo cargado (solo 1 vez por sesión)
    rows_base = df_in.fillna("").to_dict("records")
    if "down_hash" not in st.session_state:
        st.session_state["down_hash"] = _hash_rows(rows_base)

    df_in = df_in.fillna("").astype(str)

    # --- ÚNICO editor ---
    df_edit = st.data_editor(
        df_in,
        num_rows="dynamic",
        use_container_width=True,
        column_config=colcfg,
        hide_index=True,
        key="down_editor_native",
    )

    # auto-guardado
    rows = df_edit.fillna("").to_dict("records")
    h = _hash_rows(rows)
    if h != st.session_state["down_hash"]:
        guardar_sintesis_global(db, {**datos, "down_tabla": rows})
        st.session_state["down_hash"] = h
        st.toast("Guardado", icon="✅")
