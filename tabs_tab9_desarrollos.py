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

    # Subtítulo y campo para tiempo de síntesis
    st.markdown('Tiempo de síntesis')
    tiempo_sintesis = st.text_input('Tiempo de síntesis', value=st.session_state.get('tiempo_sintesis', ''), key='tiempo_sintesis', on_change=guardar_en_firestore)

    st.markdown('Perfil de temperatura')
    columnas = ['t [hora]', 't [hh:mm:ss]', 'T [°C]']
    import pandas as pd
    if 'perfil_temp_manual' not in st.session_state or list(st.session_state['perfil_temp_manual'].columns) != columnas:
        st.session_state['perfil_temp_manual'] = pd.DataFrame(
            [['' for _ in columnas] for _ in range(6)], columns=columnas
        )
    perfil_temp_manual = st.data_editor(
        st.session_state['perfil_temp_manual'],
        num_rows='fixed',
        use_container_width=True,
        key="perfil_temp_manual_editor"
    )
    st.session_state['perfil_temp_manual'] = perfil_temp_manual

    # Botón para guardar manualmente el perfil de temperatura
    if st.button('Guardar perfil de temperatura'):
        guardar_en_firestore()

    # Sección Muestreo
    st.markdown('Muestreo')
    tiempo_muestreo = st.text_input('Tiempo de muestreo', value=st.session_state.get('tiempo_muestreo', ''), key='tiempo_muestreo', on_change=guardar_en_firestore)
    tratamiento_muestras = st.text_area('Tratamiento de muestras', value=st.session_state.get('tratamiento_muestras', ''), key='tratamiento_muestras', on_change=guardar_en_firestore, height=220)

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
            "tiempo_sintesis": st.session_state.get('tiempo_sintesis', ''),
            "perfil_temperatura": st.session_state['perfil_temp_manual'].astype(str).to_dict('records'),
            "tiempo_muestreo": st.session_state.get('tiempo_muestreo', ''),
            "tratamiento_muestras": st.session_state.get('tratamiento_muestras', ''),
        }
        guardar_sintesis_global(db, datos)

    # Elimino la sección de condiciones experimentales
    # st.text_area("Condiciones experimentales (temperatura, tiempo, catalizador, etc.)", ...)
    st.text_area("Observaciones", value=st.session_state.get('observaciones', ''), key="observaciones", on_change=guardar_en_firestore)

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
    