import streamlit as st
import pandas as pd
from datetime import date, datetime
from io import BytesIO
import json
from firestore_utils import cargar_muestras, guardar_muestra
from ui_utils import mostrar_sector_flotante

def render_tab1(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Laboratorio de Polioles")
    st.session_state["current_tab"] = "Laboratorio de Polioles"
    muestras = cargar_muestras(db)
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

    st.subheader("Análisis cargados")
    muestras = cargar_muestras(db)
    tabla = []
    for m in muestras:
        for a in m["analisis"]:
            tabla.append({
                "Nombre": m["nombre"],
                "Tipo": a["tipo"],
                "Valor": a["valor"],
                "Fecha": a["fecha"],
                "Observaciones": a["observaciones"]
            })
    df_vista = pd.DataFrame(tabla)
    if not df_vista.empty:
        st.dataframe(df_vista, use_container_width=True)
        st.subheader("Eliminar análisis")
        seleccion = st.selectbox("Seleccionar análisis a eliminar", df_vista.index,
            format_func=lambda i: f"{df_vista.at[i, 'Nombre']} – {df_vista.at[i, 'Tipo']} – {df_vista.at[i, 'Fecha']}– {df_vista.at[i, 'Observaciones']}")
        if st.button("Eliminar análisis"):
            elegido = df_vista.iloc[seleccion]
            for m in muestras:
                if m["nombre"] == elegido["Nombre"]:
                    m["analisis"] = [a for a in m["analisis"] if not (
                        a["tipo"] == elegido["Tipo"] and
                        str(a["fecha"]) == elegido["Fecha"] and
                        a["valor"] == elegido["Valor"] and
                        a["observaciones"] == elegido["Observaciones"])]
                    guardar_muestra(db, m["nombre"], m["observacion"], m["analisis"], m.get("espectros", []))
                    st.success("Análisis eliminado.")
                    st.rerun()

        st.subheader("Exportar")
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_vista.to_excel(writer, index=False, sheet_name="Muestras")
        st.download_button("Descargar Excel",
            data=buffer.getvalue(),
            file_name=f"lab-polioles_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("No hay análisis cargados.")

    mostrar_sector_flotante(db, key_suffix="tab1")
