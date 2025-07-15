import streamlit as st
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global
from ui_utils import get_caracteristicas_mp, get_caracteristicas_pt


def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    CARACTERISTICAS_MP = get_caracteristicas_mp()
    CARACTERISTICAS_PT = get_caracteristicas_pt()

    # --- Cargar datos globales de síntesis al iniciar (solo una vez por sesión) ---
    if 'sintesis_global_cargada' not in st.session_state:
        datos_cargados = cargar_sintesis_global(db)
        if datos_cargados:
            for campo in ['nombre_mp', 'proveedor_mp', 'lote_mp', 'cantidad_mp', 'objetivo', 'condiciones', 'observaciones', 'downstream']:
                if campo in datos_cargados:
                    st.session_state[campo] = datos_cargados[campo]
            if 'caract_mp' in datos_cargados:
                for c in CARACTERISTICAS_MP:
                    st.session_state[f"caract_mp_{c}"] = c in datos_cargados['caract_mp']
            if 'caract_pt' in datos_cargados:
                for c in CARACTERISTICAS_PT:
                    st.session_state[f"caract_pt_{c}"] = c in datos_cargados['caract_pt']
        st.session_state['sintesis_global_cargada'] = True

    def guardar_en_firestore():
        datos = {
            "nombre_mp": st.session_state['nombre_mp'],
            "proveedor_mp": st.session_state['proveedor_mp'],
            "lote_mp": st.session_state['lote_mp'],
            "cantidad_mp": st.session_state['cantidad_mp'],
            "caract_mp": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
            "objetivo": st.session_state['objetivo'],
            "condiciones": st.session_state['condiciones'],
            "observaciones": st.session_state['observaciones'],
            "downstream": st.session_state['downstream'],
            "caract_pt": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)]
        }
        guardar_sintesis_global(db, datos)

    st.header("1. Materia Prima (MP)")
    st.text_input("Nombre de la materia prima", value=st.session_state.get('nombre_mp', ''), key="nombre_mp", on_change=guardar_en_firestore)
    st.text_input("Proveedor", value=st.session_state.get('proveedor_mp', ''), key="proveedor_mp", on_change=guardar_en_firestore)
    st.text_input("Lote", value=st.session_state.get('lote_mp', ''), key="lote_mp", on_change=guardar_en_firestore)
    st.number_input("Cantidad (g)", min_value=0.0, step=0.1, value=st.session_state.get('cantidad_mp', 0.0), key="cantidad_mp", on_change=guardar_en_firestore)

    st.header("2. Caracterización de MP")
    st.markdown("Selecciona las características a determinar:")
    cols_mp = st.columns(4)
    for idx, c in enumerate(CARACTERISTICAS_MP):
        with cols_mp[idx % 4]:
            st.checkbox(c, value=st.session_state.get(f"caract_mp_{c}", False), key=f"caract_mp_{c}", on_change=guardar_en_firestore)

    st.header("3. Síntesis")
    st.text_area("Objetivo de la síntesis", value=st.session_state.get('objetivo', ''), key="objetivo", on_change=guardar_en_firestore)
    st.text_area("Condiciones experimentales (temperatura, tiempo, catalizador, etc.)", value=st.session_state.get('condiciones', ''), key="condiciones", on_change=guardar_en_firestore)
    st.text_area("Observaciones adicionales", value=st.session_state.get('observaciones', ''), key="observaciones", on_change=guardar_en_firestore)

    st.header("4. Downstream (Procesos posteriores)")
    st.text_area("Descripción de procesos downstream (purificación, separación, etc.)", value=st.session_state.get('downstream', ''), key="downstream", on_change=guardar_en_firestore)

    st.header("5. Caracterización del Producto (PT)")
    st.markdown("Selecciona las características a determinar en el producto terminado:")
    cols_pt = st.columns(4)
    for idx, c in enumerate(CARACTERISTICAS_PT):
        with cols_pt[idx % 4]:
            st.checkbox(c, value=st.session_state.get(f"caract_pt_{c}", False), key=f"caract_pt_{c}", on_change=guardar_en_firestore)

    st.write("**Resumen de la síntesis:**")
    st.write({
        "Materia Prima": {
            "Nombre": st.session_state.get('nombre_mp', ''),
            "Proveedor": st.session_state.get('proveedor_mp', ''),
            "Lote": st.session_state.get('lote_mp', ''),
            "Cantidad": st.session_state.get('cantidad_mp', 0.0)
        },
        "Características MP": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
        "Síntesis": {
            "Objetivo": st.session_state.get('objetivo', ''),
            "Condiciones": st.session_state.get('condiciones', ''),
            "Observaciones": st.session_state.get('observaciones', '')
        },
        "Downstream": st.session_state.get('downstream', ''),
        "Características PT": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)]
    })
    