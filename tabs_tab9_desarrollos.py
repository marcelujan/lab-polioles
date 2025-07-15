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
            # Elimino los campos de materia prima que ya no se usan
            "caract_mp": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
            "observaciones_mp": st.session_state.get('observaciones_mp', ''),
            "objetivo": st.session_state['objetivo'],
            "condiciones": st.session_state['condiciones'],
            "observaciones": st.session_state['observaciones'],
            "downstream": st.session_state['downstream'],
            "caract_pt": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)]
        }
        guardar_sintesis_global(db, datos)

    # Muevo el campo de objetivo de la síntesis al principio
    st.text_area("Objetivo de la síntesis", value=st.session_state.get('objetivo', ''), key="objetivo", on_change=guardar_en_firestore)

    st.header("01 MP")
    st.text_input("Aceite de soja", value=st.session_state.get('aceite_soja', ''), key="aceite_soja", on_change=guardar_en_firestore, placeholder="Especificar tipo o marca de aceite de soja...")

    st.header("02 CARACT MP")
    st.markdown("Aceite de soja")
    cols_mp = st.columns(4)
    for idx, c in enumerate(CARACTERISTICAS_MP):
        with cols_mp[idx % 4]:
            st.checkbox(c, value=st.session_state.get(f"caract_mp_{c}", False), key=f"caract_mp_{c}", on_change=guardar_en_firestore)

    observaciones_mp = st.text_area("Observaciones", value=st.session_state.get('observaciones_mp', ''), key="observaciones_mp", on_change=guardar_en_firestore)

    st.header("03 SÍNTESIS")

    # Tabla manual de perfil de temperatura
    columnas_editables = ['t inicial', 'tf-ti', 'T [°C] inicial', 'T [°C] final']
    columnas_finales = ['t inicial', 'tf-ti', 't final', 'T [°C] inicial', 'T [°C] final']
    import pandas as pd
    if 'perfil_temp_manual' not in st.session_state:
        st.session_state['perfil_temp_manual'] = pd.DataFrame(
            [['' for _ in columnas_editables] for _ in range(6)], columns=columnas_editables
        )
    # Editor solo para columnas editables
    perfil_temp_manual = st.data_editor(
        st.session_state['perfil_temp_manual'],
        num_rows=6,
        use_container_width=True,
        key="perfil_temp_manual_editor"
    )
    st.session_state['perfil_temp_manual'] = perfil_temp_manual

    # Calcular t final
    from datetime import datetime, timedelta
    tabla_resultado = []
    for idx, row in perfil_temp_manual.iterrows():
        t_ini = row.get('t inicial', '')
        tf_ti = row.get('tf-ti', '')
        t_ini_str = t_ini.strip() if isinstance(t_ini, str) else ''
        tf_ti_str = tf_ti.strip() if isinstance(tf_ti, str) else ''
        t_final = ''
        error = False
        if t_ini_str and tf_ti_str:
            try:
                t_ini_dt = datetime.strptime(t_ini_str, "%H:%M:%S")
                tf_ti_dt = datetime.strptime(tf_ti_str, "%H:%M:%S")
                delta = timedelta(hours=tf_ti_dt.hour, minutes=tf_ti_dt.minute, seconds=tf_ti_dt.second)
                t_final_dt = t_ini_dt + delta
                t_final = t_final_dt.strftime("%H:%M:%S")
            except Exception:
                t_final = '⚠️ Error formato'
                error = True
        tabla_resultado.append({
            't inicial': t_ini_str,
            'tf-ti': tf_ti_str,
            't final': t_final,
            'T [°C] inicial': row.get('T [°C] inicial', ''),
            'T [°C] final': row.get('T [°C] final', '')
        })
    st.markdown("**Tabla con t final calculado automáticamente:**")
    st.dataframe(pd.DataFrame(tabla_resultado, columns=columnas_finales), use_container_width=True)

    # Guardar en Firestore al modificar
    def guardar_en_firestore():
        datos = {
            "caract_mp": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
            "observaciones_mp": st.session_state.get('observaciones_mp', ''),
            "objetivo": st.session_state['objetivo'],
            "condiciones": st.session_state['condiciones'],
            "observaciones": st.session_state['observaciones'],
            "downstream": st.session_state['downstream'],
            "caract_pt": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)],
            "perfil_temperatura": tabla_resultado
        }
        guardar_sintesis_global(db, datos)

    st.text_area("Condiciones experimentales (temperatura, tiempo, catalizador, etc.)", value=st.session_state.get('condiciones', ''), key="condiciones", on_change=guardar_en_firestore)
    st.text_area("Observaciones adicionales", value=st.session_state.get('observaciones', ''), key="observaciones", on_change=guardar_en_firestore)

    st.header("DOWNSTREAM")
    st.text_area("Descripción de procesos downstream (purificación, separación, etc.)", value=st.session_state.get('downstream', ''), key="downstream", on_change=guardar_en_firestore)

    st.header("09 CARACT PT")
    st.markdown("Selecciona las características a determinar en el producto terminado:")
    cols_pt = st.columns(4)
    for idx, c in enumerate(CARACTERISTICAS_PT):
        with cols_pt[idx % 4]:
            st.checkbox(c, value=st.session_state.get(f"caract_pt_{c}", False), key=f"caract_pt_{c}", on_change=guardar_en_firestore)

    st.write("**Resumen de la síntesis:**")
    st.write({
        # Elimino la sección de Materia Prima
        "Características MP": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
        "Observaciones MP": st.session_state.get('observaciones_mp', ''),
        "Síntesis": {
            "Objetivo": st.session_state.get('objetivo', ''),
            "Condiciones": st.session_state.get('condiciones', ''),
            "Observaciones": st.session_state.get('observaciones', '')
        },
        "Downstream": st.session_state.get('downstream', ''),
        "Características PT": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)]
    })
    