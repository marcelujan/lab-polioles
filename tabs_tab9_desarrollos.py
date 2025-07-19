import streamlit as st
import pandas as pd
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global
from ui_utils import get_caracteristicas_mp, get_caracteristicas_pt


def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    CARACTERISTICAS_MP = get_caracteristicas_mp()
    CARACTERISTICAS_PT = get_caracteristicas_pt()

    # --- Cargar datos globales de síntesis al iniciar (solo una vez por sesión) ---
    if 'sintesis_global_cargada' not in st.session_state:
        datos_cargados = cargar_sintesis_global(db)
        if datos_cargados:
            # Campos básicos
            for campo in ['nombre_mp', 'proveedor_mp', 'lote_mp', 'cantidad_mp', 'objetivo', 'condiciones', 'observaciones', 'downstream']:
                if campo in datos_cargados:
                    st.session_state[campo] = datos_cargados[campo]
            
            # Campo aceite_soja
            if 'aceite_soja' in datos_cargados:
                st.session_state['aceite_soja'] = datos_cargados['aceite_soja']
            
            # Campo observaciones_mp
            if 'observaciones_mp' in datos_cargados:
                st.session_state['observaciones_mp'] = datos_cargados['observaciones_mp']
            
            # Campo tiempo_sintesis
            if 'tiempo_sintesis' in datos_cargados:
                st.session_state['tiempo_sintesis'] = datos_cargados['tiempo_sintesis']
            
            # Campo tiempo_muestreo
            if 'tiempo_muestreo' in datos_cargados:
                st.session_state['tiempo_muestreo'] = datos_cargados['tiempo_muestreo']
            
            # Campo tratamiento_muestras
            if 'tratamiento_muestras' in datos_cargados:
                st.session_state['tratamiento_muestras'] = datos_cargados['tratamiento_muestras']
            
            # Características MP
            if 'caract_mp' in datos_cargados:
                for c in CARACTERISTICAS_MP:
                    st.session_state[f"caract_mp_{c}"] = c in datos_cargados['caract_mp']
            
            # Características PT
            if 'caract_pt' in datos_cargados:
                for c in CARACTERISTICAS_PT:
                    st.session_state[f"caract_pt_{c}"] = c in datos_cargados['caract_pt']
            
            # Perfil de temperatura
            if 'perfil_temperatura' in datos_cargados:
                st.write("🔍 Debug - Cargando perfil de temperatura desde Firestore:")
                st.write(f"Datos cargados: {datos_cargados['perfil_temperatura']}")
                
                try:
                    # Intentar cargar el DataFrame desde los datos guardados
                    perfil_data = datos_cargados['perfil_temperatura']
                    if perfil_data and len(perfil_data) > 0:
                        # Convertir de vuelta a DataFrame
                        df_temp = pd.DataFrame(perfil_data)
                        # Asegurar que tenga las columnas correctas
                        if list(df_temp.columns) == ['t [hora]', 't [hh:mm:ss]', 'T [°C]']:
                            st.session_state['perfil_temp_manual'] = df_temp
                            st.success("✅ Perfil de temperatura cargado correctamente")
                        else:
                            # Si las columnas no coinciden, crear DataFrame vacío
                            data = [['', '', ''] for _ in range(6)]
                            st.session_state['perfil_temp_manual'] = pd.DataFrame(
                                data, 
                                columns=['t [hora]', 't [hh:mm:ss]', 'T [°C]']
                            )
                            st.warning("⚠️ Columnas no coinciden, creando tabla vacía")
                    else:
                        # Si no hay datos, crear DataFrame vacío
                        data = [['', '', ''] for _ in range(6)]
                        st.session_state['perfil_temp_manual'] = pd.DataFrame(
                            data, 
                            columns=['t [hora]', 't [hh:mm:ss]', 'T [°C]']
                        )
                        st.info("ℹ️ No hay datos de perfil guardados, creando tabla vacía")
                except Exception as e:
                    st.error(f"❌ Error al cargar perfil de temperatura: {e}")
                    # Si hay error, crear DataFrame vacío
                    data = [['', '', ''] for _ in range(6)]
                    st.session_state['perfil_temp_manual'] = pd.DataFrame(
                        data, 
                        columns=['t [hora]', 't [hh:mm:ss]', 'T [°C]']
                    )
        st.session_state['sintesis_global_cargada'] = True

    def guardar_en_firestore():
        datos = {
            # Campos básicos
            "caract_mp": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
            "observaciones_mp": st.session_state.get('observaciones_mp', ''),
            "objetivo": st.session_state.get('objetivo', ''),
            "condiciones": st.session_state.get('condiciones', ''),
            "observaciones": st.session_state.get('observaciones', ''),
            "downstream": st.session_state.get('downstream', ''),
            "caract_pt": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)],
            
            # Campos adicionales
            "aceite_soja": st.session_state.get('aceite_soja', ''),
            "tiempo_sintesis": st.session_state.get('tiempo_sintesis', ''),
            "tiempo_muestreo": st.session_state.get('tiempo_muestreo', ''),
            "tratamiento_muestras": st.session_state.get('tratamiento_muestras', ''),
        }
        
        # Agregar perfil de temperatura si existe
        if 'perfil_temp_manual' in st.session_state:
            datos["perfil_temperatura"] = st.session_state['perfil_temp_manual'].astype(str).to_dict('records')
        
        guardar_sintesis_global(db, datos)

    # Muevo el campo de objetivo de la síntesis al principio
    st.text_area("Objetivo de la síntesis", key="objetivo", on_change=guardar_en_firestore)

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

    # Subtítulo 
    st.markdown('Perfil de temperatura')
    if 'perfil_temp_manual' not in st.session_state or list(st.session_state['perfil_temp_manual'].columns) != ['t [hora]', 't [hh:mm:ss]', 'T [°C]']:
        data = [['', '', ''] for _ in range(6)]
        st.session_state['perfil_temp_manual'] = pd.DataFrame(
            data, 
            columns=['t [hora]', 't [hh:mm:ss]', 'T [°C]']
        )
    
    # Función para guardar automáticamente cuando cambie la tabla
    def guardar_perfil_temp():
        if 'perfil_temp_manual' in st.session_state:
            # Debug: mostrar qué se está guardando
            st.write("🔍 Debug - Guardando perfil de temperatura:")
            st.write(f"Datos de la tabla: {st.session_state['perfil_temp_manual'].to_dict('records')}")
            
            datos = {
                "caract_mp": [c for c in CARACTERISTICAS_MP if st.session_state.get(f"caract_mp_{c}", False)],
                "observaciones_mp": st.session_state.get('observaciones_mp', ''),
                "objetivo": st.session_state.get('objetivo', ''),
                "condiciones": st.session_state.get('condiciones', ''),
                "observaciones": st.session_state.get('observaciones', ''),
                "downstream": st.session_state.get('downstream', ''),
                "caract_pt": [c for c in CARACTERISTICAS_PT if st.session_state.get(f"caract_pt_{c}", False)],
                "aceite_soja": st.session_state.get('aceite_soja', ''),
                "tiempo_sintesis": st.session_state.get('tiempo_sintesis', ''),
                "tiempo_muestreo": st.session_state.get('tiempo_muestreo', ''),
                "tratamiento_muestras": st.session_state.get('tratamiento_muestras', ''),
                "perfil_temperatura": st.session_state['perfil_temp_manual'].astype(str).to_dict('records')
            }
            guardar_sintesis_global(db, datos)
            st.success("✅ Perfil guardado en Firestore")
    
    perfil_temp_manual = st.data_editor(
        st.session_state['perfil_temp_manual'],
        num_rows='fixed',
        use_container_width=True,
        key="perfil_temp_manual_editor",
        on_change=guardar_perfil_temp
    )
    st.session_state['perfil_temp_manual'] = perfil_temp_manual

    # Botón para guardar manualmente el perfil de temperatura (opcional)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button('💾 Guardar perfil de temperatura'):
            guardar_perfil_temp()
            st.success("Perfil de temperatura guardado!")
    with col2:
        st.info("💡 La tabla se guarda automáticamente al editar. El botón es opcional.")

    # Sección Muestreo
    st.markdown('Muestreo')
    # Solo una vez el campo 'Tiempo de síntesis', dentro de Muestreo
    tiempo_sintesis = st.text_input('Tiempo de síntesis', key='tiempo_sintesis', on_change=guardar_en_firestore)
    tiempo_muestreo = st.text_input('Tiempo de muestreo', value=st.session_state.get('tiempo_muestreo', ''), key='tiempo_muestreo', on_change=guardar_en_firestore)
    tratamiento_muestras = st.text_area('Tratamiento de muestras', value=st.session_state.get('tratamiento_muestras', ''), key='tratamiento_muestras', on_change=guardar_en_firestore, height=220)

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
    