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

    # Campo para hora de inicio
    hora_inicio = st.time_input("Hora de inicio", value=st.session_state.get('hora_inicio', None) or None, key="hora_inicio", help="Hora de inicio del perfil de temperatura")

    # Tabla editable tipo Excel
    if 'perfil_temp_edit' not in st.session_state:
        st.session_state['perfil_temp_edit'] = [
            {"t_hora": "00:00:00", "T_C": 25},
            {"t_hora": "00:30:00", "T_C": 35},
            {"t_hora": "01:00:00", "T_C": 45},
        ]
    import pandas as pd
    # Asegura que siempre sea DataFrame al editar
    if isinstance(st.session_state['perfil_temp_edit'], list):
        perfil_temp_df = pd.DataFrame(st.session_state['perfil_temp_edit'])
    else:
        perfil_temp_df = st.session_state['perfil_temp_edit']

    perfil_temp_df = st.data_editor(
        perfil_temp_df,
        num_rows="dynamic",
        use_container_width=True,
        key="perfil_temp_editor"
    )
    # Guarda siempre como lista de dicts
    st.session_state['perfil_temp_edit'] = perfil_temp_df.to_dict("records")

    # Calcular tabla resultante
    from datetime import datetime, timedelta
    def sumar_horas(hora_base, delta_str):
        h, m, s = map(int, delta_str.split(":"))
        return (hora_base + timedelta(hours=h, minutes=m, seconds=s)).strftime("%H:%M:%S")

    tabla_resultado = []
    if hora_inicio is not None and st.session_state['perfil_temp_edit']:
        hora_base = datetime.strptime(str(hora_inicio), "%H:%M:%S")
        temp_anterior = None
        hora_actual = hora_base
        for idx, fila in enumerate(st.session_state['perfil_temp_edit']):
            t_hora = fila.get("t_hora", "00:00:00")
            T_C = fila.get("T_C", "")
            if idx == 0:
                rango_temp = str(T_C)
            else:
                rango_temp = f"{temp_anterior} --> {T_C}"
            hora_absoluta = sumar_horas(hora_base, t_hora)
            tabla_resultado.append({
                "t [hora]": t_hora,
                "t [hh:mm:ss]": hora_absoluta,
                "T [°C]": rango_temp
            })
            temp_anterior = T_C
    st.markdown("**Tabla calculada de perfil de temperatura:**")
    st.dataframe(pd.DataFrame(tabla_resultado), use_container_width=True)

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
            "perfil_temperatura": {
                "hora_inicio": str(hora_inicio) if hora_inicio else None,
                "tabla": st.session_state['perfil_temp_edit']
            }
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
    