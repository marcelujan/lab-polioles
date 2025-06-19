import streamlit as st
import pandas as pd
from datetime import date, datetime
from io import BytesIO
import json
from firestore_utils import cargar_muestras, guardar_muestra
from ui_utils import mostrar_sector_flotante
from firestore_utils import eliminar_muestra  # asegurate de tener esta línea al inicio

def render_tab1(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Laboratorio de Polioles")
    st.session_state["current_tab"] = "Laboratorio de Polioles"
    muestras = cargar_muestras(db)

    if st.checkbox("Mostrar resumen de observaciones", key="mostrar_resumen_obs"):
        st.markdown("#### 📝 Selecciona muestras para ver sus observaciones:")
        
        observaciones_seleccionadas = []

        for m in muestras:
            nombre = m.get("nombre", "Sin nombre")
            obs = m.get("observacion", "")
            key_checkbox = f"ver_obs_{nombre}"

            if st.checkbox(nombre, key=key_checkbox):
                observaciones_seleccionadas.append(f"🔹 **{nombre}**:\n{obs.strip()}")

        if observaciones_seleccionadas:
            st.markdown("#### 🧾 Observaciones combinadas:")
            st.markdown(
                "<div style='white-space: pre-wrap; border: 1px solid #ccc; padding: 1em; border-radius: 10px;'>"
                + "\n\n---\n\n".join(observaciones_seleccionadas)
                + "</div>",
                unsafe_allow_html=True
            )

    st.subheader("Añadir muestra")
    nombres = [m["nombre"] for m in muestras]

    opcion = st.selectbox("Seleccionar muestra", ["Nueva muestra"] + nombres)
    if opcion == "Nueva muestra":
        nombre_muestra = st.text_input("Nombre de nueva muestra")
        muestra_existente = None
    else:
        nombre_muestra = opcion
        muestra_existente = next((m for m in muestras if m["nombre"] == opcion), None)
    st.session_state["muestra_activa"] = nombre_muestra

    observacion = st.text_area("Observaciones", value=muestra_existente["observacion"] if muestra_existente else "", height=150)

    st.subheader("Nuevo análisis")
    tipos = [
        "Índice de yodo [% p/p I2 abs]", "Índice OH [mg KHO/g]",
        "Índice de acidez [mg KOH/g]", "Índice de epóxido [mol/100g]",
        "Humedad [%]", "PM [g/mol]", "Funcionalidad [#]",
        "Viscosidad dinámica [cP]", "Densidad [g/mL]", "Otro análisis"
    ]
    df = pd.DataFrame([{"Tipo": "", "Valor": 0.0, "Fecha": date.today(), "Observaciones": ""}])
    nuevos_analisis = st.data_editor(df, num_rows="dynamic", use_container_width=True,
        column_config={"Tipo": st.column_config.SelectboxColumn("Tipo", options=tipos)})

    if st.button("Guardar análisis"):
        previos = muestra_existente["analisis"] if muestra_existente else []
        nuevos = []
        for _, row in nuevos_analisis.iterrows():
            if row["Tipo"] != "":
                tipo = row["Tipo"]
                valor = row["Valor"]
                fecha = str(row["Fecha"])
                obs = row["Observaciones"]
                resumen_obs = obs.replace("\n", " ").strip()[:30].replace(" ", "_")
                id_unico = f"{tipo}-{valor}-{fecha}-{resumen_obs}"
                nuevos.append({
                    "tipo": tipo,
                    "valor": valor,
                    "fecha": fecha,
                    "observaciones": obs,
                    "id": id_unico
                })
        nuevos_validos = [a for a in nuevos if a["tipo"] != "" and a["valor"] != 0]
        guardar_muestra(
            db,
            nombre_muestra,
            observacion,
            previos + nuevos_validos,
            muestra_existente.get("espectros") if muestra_existente else []
        )
        st.success("Análisis guardado.")
        st.rerun()


    muestras = cargar_muestras(db)
    tabla = []
    for m in muestras:
        for a in m.get("analisis", []):
            tabla.append({
                "Nombre": m.get("nombre", ""),
                "Tipo": a.get("tipo", ""),
                "Valor": a.get("valor", ""),
                "Fecha": a.get("fecha", ""),
                "Observaciones": a.get("observaciones", "")
            })

    df_vista = pd.DataFrame(tabla)
    if not df_vista.empty:
        st.dataframe(df_vista, use_container_width=True)
        st.subheader("Eliminar análisis")
        seleccion = st.selectbox("Seleccionar análisis a eliminar", df_vista.index,
            format_func=lambda i: f"{df_vista.at[i, 'Nombre']} – {df_vista.at[i, 'Tipo']} – {df_vista.at[i, 'Fecha']}– {df_vista.at[i, 'Observaciones']}")
        confirmacion_analisis = st.checkbox("Confirmar eliminación del análisis seleccionado", key="confirmar_borrado_analisis")
        if st.button("Eliminar análisis"):
            if confirmacion_analisis:
                elegido = df_vista.iloc[seleccion]
                for m in muestras:
                    if m["nombre"] == elegido["Nombre"]:
                        m["analisis"] = [a for a in m.get("analisis", []) if not (
                            str(a.get("tipo", "")) == str(elegido["Tipo"]) and
                            str(a.get("fecha", "")) == str(elegido["Fecha"]) and
                            str(a.get("valor", "")) == str(elegido["Valor"]) and
                            str(a.get("observaciones", "")) == str(elegido["Observaciones"])
                        )]
                        guardar_muestra(db, m["nombre"], m.get("observacion", ""), m["analisis"], m.get("espectros", []))
                        st.success("Análisis eliminado.")
                        st.rerun()
            else:
                st.warning("Debes marcar la casilla de confirmación para eliminar el análisis.")


    # Eliminar muestra completa
    st.subheader("Eliminar muestra")
    nombres_muestras = sorted(set(m["nombre"] for m in muestras))
    muestra_a_borrar = st.selectbox("Seleccionar muestra a eliminar", nombres_muestras)

    confirmacion = st.checkbox(f"Confirmar eliminación de '{muestra_a_borrar}'", key="confirmar_borrado_muestra")

    if st.button("Eliminar muestra"):
        if confirmacion:
            st.info(f"Intentando eliminar documento: {muestra_a_borrar}")
            try:
                ref = db.collection("muestras").document(muestra_a_borrar)
                if ref.get().exists:
                    st.warning("Documento encontrado. Procediendo a eliminar...")
                    ref.delete()
                    st.success(f"✅ Documento '{muestra_a_borrar}' eliminado.")
                else:
                    st.info("⚠ El documento ya no existe.")
            except Exception as e:
                st.error(f"❌ Error al intentar eliminar la muestra: {e}")
            st.rerun()
        else:
            st.warning("Debes marcar la casilla de confirmación para eliminar la muestra.")

    # Descargar excel
    st.subheader("Exportar")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_vista.to_excel(writer, index=False, sheet_name="Muestras")
    st.download_button("Descargar Excel",
        data=buffer.getvalue(),
        file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    mostrar_sector_flotante(db, key_suffix="tab1")
