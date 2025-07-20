import streamlit as st
import pandas as pd
from datetime import date, datetime
from io import BytesIO
import json
from firestore_utils import cargar_muestras, guardar_muestra
from firestore_utils import eliminar_muestra  # asegurate de tener esta línea al inicio
from ui_utils import get_caracteristicas

def render_tab1(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Laboratorio de Polioles")
    st.session_state["current_tab"] = "Laboratorio de Polioles"
    muestras = cargar_muestras(db)

    if st.checkbox("Mostrar resumen de observaciones", key="mostrar_resumen_obs"):

        muestras = cargar_muestras(db)  # actualizar

        # Parte 1: checkboxes en 4 columnas
        muestras_seleccionadas = []
        cols = st.columns(4)

        for idx, m in enumerate(muestras):
            nombre = m.get("nombre", "Sin nombre")
            key_checkbox = f"ver_obs_{nombre}"
            with cols[idx % 4]:
                if st.checkbox(nombre, key=key_checkbox):
                    muestras_seleccionadas.append(nombre)

        # Parte 2: edición de observaciones (en orden de selección)
        observaciones_modificadas = {}

        for nombre in muestras_seleccionadas:
            muestra = next((m for m in muestras if m["nombre"] == nombre), None)
            if muestra:
                obs_original = muestra.get("observacion", "")
                texto_prellenado = f"{nombre}: {obs_original.strip()}" if obs_original.strip() else f"{nombre}: "

                key_textarea = f"textarea_obs_{nombre}"
                nueva_obs = st.text_area(
                    label=f"Observación {nombre}",
                    value=texto_prellenado,
                    key=key_textarea,
                    height=70,
                    label_visibility="collapsed"
                )

                # Remover el prefijo "Nombre: " para guardar solo la observación
                prefijo = f"{nombre}:"
                if nueva_obs.startswith(prefijo):
                    solo_obs = nueva_obs[len(prefijo):].strip()
                else:
                    solo_obs = nueva_obs.strip()

                if solo_obs != obs_original.strip():
                    observaciones_modificadas[nombre] = solo_obs

        if observaciones_modificadas:
            if st.button("💾 Guardar cambios en observaciones"):
                muestras = cargar_muestras(db)  # recargar por seguridad
                for m in muestras:
                    nombre = m["nombre"]
                    if nombre in observaciones_modificadas:
                        guardar_muestra(
                            db,
                            nombre,
                            observaciones_modificadas[nombre],
                            m.get("analisis", []),
                            m.get("espectros", [])
                        )
                st.success("✔ Observaciones actualizadas.")
                st.rerun()

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

