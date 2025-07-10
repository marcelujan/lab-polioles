import streamlit as st

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

    with st.form("form_sintesis"):
        st.header("1. Materia Prima (MP)")
        nombre_mp = st.text_input("Nombre de la materia prima")
        proveedor_mp = st.text_input("Proveedor")
        lote_mp = st.text_input("Lote")
        cantidad_mp = st.number_input("Cantidad (g)", min_value=0.0, step=0.1)

        st.header("2. Caracterización de MP")
        st.markdown("Selecciona las características a determinar:")
        caract_mp = {}
        for c in CARACTERISTICAS_MP:
            caract_mp[c] = st.checkbox(c, key=f"caract_mp_{c}")

        st.header("3. Síntesis")
        objetivo = st.text_area("Objetivo de la síntesis")
        condiciones = st.text_area("Condiciones experimentales (temperatura, tiempo, catalizador, etc.)")
        observaciones = st.text_area("Observaciones adicionales")

        st.header("4. Downstream (Procesos posteriores)")
        downstream = st.text_area("Descripción de procesos downstream (purificación, separación, etc.)")

        st.header("5. Caracterización del Producto (PT)")
        st.markdown("Selecciona las características a determinar en el producto terminado:")
        caract_pt = {}
        for c in CARACTERISTICAS_PT:
            caract_pt[c] = st.checkbox(c, key=f"caract_pt_{c}")

        submitted = st.form_submit_button("Guardar síntesis")

    if submitted:
        st.success("¡Síntesis registrada! (Funcionalidad de guardado a implementar)")
        st.write("**Resumen de la síntesis:**")
        st.write({
            "Materia Prima": {
                "Nombre": nombre_mp,
                "Proveedor": proveedor_mp,
                "Lote": lote_mp,
                "Cantidad": cantidad_mp
            },
            "Características MP": [k for k, v in caract_mp.items() if v],
            "Síntesis": {
                "Objetivo": objetivo,
                "Condiciones": condiciones,
                "Observaciones": observaciones
            },
            "Downstream": downstream,
            "Características PT": [k for k, v in caract_pt.items() if v]
        })
    