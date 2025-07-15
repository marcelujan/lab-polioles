import streamlit as st
from firestore_utils import cargar_sintesis_global, guardar_sintesis_global

# Opciones de características para checkboxes
CARACTERISTICAS_MP = [
    'Índice OH', 'Índice de yodo', 'Índice ácido', 'Índice de peróxidos', 'Humedad', 'Color', 'Viscosidad'
]
CARACTERISTICAS_PT = [
    'Densidad', 'Viscosidad', 'Índice OH', 'Índice de yodo', 'Índice ácido', 'Color', 'Estabilidad', 'Solubilidad'
]


def render_tab9(db, cargar_muestras, mostrar_sector_flotante):
    st.title("Desarrollos de Síntesis")
    st.markdown("""
    <span style='font-size:1.1rem;'>
    Aquí puedes pactar y registrar las características de las síntesis a realizar, para no olvidar ningún detalle importante.
    </span>
    """, unsafe_allow_html=True)

    # Inicializar estado de sesión para cada campo si no existe
    if 'nombre_mp' not in st.session_state:
        st.session_state['nombre_mp'] = ''
    if 'proveedor_mp' not in st.session_state:
        st.session_state['proveedor_mp'] = ''
    if 'lote_mp' not in st.session_state:
        st.session_state['lote_mp'] = ''
    if 'cantidad_mp' not in st.session_state:
        st.session_state['cantidad_mp'] = 0.0
    for c in CARACTERISTICAS_MP:
        if f"caract_mp_{c}" not in st.session_state:
            st.session_state[f"caract_mp_{c}"] = False
    if 'objetivo' not in st.session_state:
        st.session_state['objetivo'] = ''
    if 'condiciones' not in st.session_state:
        st.session_state['condiciones'] = ''
    if 'observaciones' not in st.session_state:
        st.session_state['observaciones'] = ''
    if 'downstream' not in st.session_state:
        st.session_state['downstream'] = ''
    for c in CARACTERISTICAS_PT:
        if f"caract_pt_{c}" not in st.session_state:
            st.session_state[f"caract_pt_{c}"] = False

    # --- Cargar datos globales de síntesis al iniciar (solo una vez por sesión) ---
    if 'sintesis_global_cargada' not in st.session_state:
        datos_cargados = cargar_sintesis_global(db)
        if datos_cargados:
            # Cargar campos de texto
            for campo in ['nombre_mp', 'proveedor_mp', 'lote_mp', 'cantidad_mp', 'objetivo', 'condiciones', 'observaciones', 'downstream']:
                if campo in datos_cargados:
                    st.session_state[campo] = datos_cargados[campo]
            # Cargar checkboxes MP
            if 'caract_mp' in datos_cargados:
                for c in CARACTERISTICAS_MP:
                    st.session_state[f"caract_mp_{c}"] = c in datos_cargados['caract_mp']
            # Cargar checkboxes PT
            if 'caract_pt' in datos_cargados:
                for c in CARACTERISTICAS_PT:
                    st.session_state[f"caract_pt_{c}"] = c in datos_cargados['caract_pt']
        st.session_state['sintesis_global_cargada'] = True

    # --- Definir función para guardar automáticamente en Firestore ---
    def guardar_en_firestore():
        datos = {
            "nombre_mp": st.session_state['nombre_mp'],
            "proveedor_mp": st.session_state['proveedor_mp'],
            "lote_mp": st.session_state['lote_mp'],
            "cantidad_mp": st.session_state['cantidad_mp'],
            "caract_mp": [c for c in CARACTERISTICAS_MP if st.session_state[f"caract_mp_{c}"]],
            "objetivo": st.session_state['objetivo'],
            "condiciones": st.session_state['condiciones'],
            "observaciones": st.session_state['observaciones'],
            "downstream": st.session_state['downstream'],
            "caract_pt": [c for c in CARACTERISTICAS_PT if st.session_state[f"caract_pt_{c}"]]
        }
        guardar_sintesis_global(db, datos)

    # --- Inputs con callbacks para guardar automáticamente ---
    st.header("1. Materia Prima (MP)")
    st.text_input("Nombre de la materia prima", value=st.session_state['nombre_mp'], key="nombre_mp", on_change=guardar_en_firestore)
    st.text_input("Proveedor", value=st.session_state['proveedor_mp'], key="proveedor_mp", on_change=guardar_en_firestore)
    st.text_input("Lote", value=st.session_state['lote_mp'], key="lote_mp", on_change=guardar_en_firestore)
    st.number_input("Cantidad (g)", min_value=0.0, step=0.1, value=st.session_state['cantidad_mp'], key="cantidad_mp", on_change=guardar_en_firestore)

    st.header("2. Caracterización de MP")
    st.markdown("Selecciona las características a determinar:")
    cols_mp = st.columns(4)
    for idx, c in enumerate(CARACTERISTICAS_MP):
        with cols_mp[idx % 4]:
            st.checkbox(c, value=st.session_state[f"caract_mp_{c}"], key=f"caract_mp_{c}", on_change=guardar_en_firestore)

    st.header("3. Síntesis")
    st.text_area("Objetivo de la síntesis", value=st.session_state['objetivo'], key="objetivo", on_change=guardar_en_firestore)
    st.text_area("Condiciones experimentales (temperatura, tiempo, catalizador, etc.)", value=st.session_state['condiciones'], key="condiciones", on_change=guardar_en_firestore)
    st.text_area("Observaciones adicionales", value=st.session_state['observaciones'], key="observaciones", on_change=guardar_en_firestore)

    st.header("4. Downstream (Procesos posteriores)")
    st.text_area("Descripción de procesos downstream (purificación, separación, etc.)", value=st.session_state['downstream'], key="downstream", on_change=guardar_en_firestore)

    st.header("5. Caracterización del Producto (PT)")
    st.markdown("Selecciona las características a determinar en el producto terminado:")
    cols_pt = st.columns(4)
    for idx, c in enumerate(CARACTERISTICAS_PT):
        with cols_pt[idx % 4]:
            st.checkbox(c, value=st.session_state[f"caract_pt_{c}"], key=f"caract_pt_{c}", on_change=guardar_en_firestore)

    # Mostrar resumen en tiempo real
    st.write("**Resumen de la síntesis:**")
    st.write({
        "Materia Prima": {
            "Nombre": st.session_state['nombre_mp'],
            "Proveedor": st.session_state['proveedor_mp'],
            "Lote": st.session_state['lote_mp'],
            "Cantidad": st.session_state['cantidad_mp']
        },
        "Características MP": [c for c in CARACTERISTICAS_MP if st.session_state[f"caract_mp_{c}"]],
        "Síntesis": {
            "Objetivo": st.session_state['objetivo'],
            "Condiciones": st.session_state['condiciones'],
            "Observaciones": st.session_state['observaciones']
        },
        "Downstream": st.session_state['downstream'],
        "Características PT": [c for c in CARACTERISTICAS_PT if st.session_state[f"caract_pt_{c}"]]
    })
    