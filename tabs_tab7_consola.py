# tabs_tab7_consola.py
import streamlit as st
import pandas as pd
import base64
from io import BytesIO
import os
import zipfile
from datetime import datetime
from tempfile import TemporaryDirectory

def render_tab7(db, cargar_muestras, guardar_muestra, mostrar_sector_flotante):
    st.title("Consola")
    st.session_state["current_tab"] = "Consola"
    muestras = cargar_muestras(db)
    if not muestras:
        st.info("No hay muestras cargadas.")
        st.stop()

    for muestra in muestras:
        with st.expander(f"📁 {muestra['nombre']}"):
            st.markdown(f"📝 **Observación:** {muestra.get('observacion', '—')}")
            analisis = muestra.get("analisis", [])
            if analisis:
                st.markdown("📊 **Análisis cargados:**")
                for a in analisis:
                    st.markdown(f"- {a['tipo']}: {a['valor']} ({a['fecha']})")
                df_analisis = pd.DataFrame(analisis)
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_analisis.to_excel(writer, index=False, sheet_name="Análisis")
                buffer.seek(0)
                st.download_button("⬇️ Descargar análisis",
                    data=buffer.getvalue(),
                    file_name=f"analisis_{muestra['nombre']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            espectros = muestra.get("espectros", [])
            if espectros:
                st.markdown("🧪 **Espectros cargados:**")
                for e in espectros:
                    etiqueta = f"{e['tipo']} ({e['fecha']})"
                    st.markdown(f"🖼️ {etiqueta}" if e.get("es_imagen", False) else f"📈 {etiqueta}")

                filas_mascaras = []
                for e in espectros:
                    if e.get("mascaras"):
                        for j, mascara in enumerate(e["mascaras"]):
                            filas_mascaras.append({
                                "Archivo": e.get("nombre_archivo", ""),
                                "Máscara N°": j + 1,
                                "D [m2/s]": mascara.get("difusividad"),
                                "T2 [s]": mascara.get("t2"),
                                "Xmin [ppm]": mascara.get("x_min"),
                                "Xmax [ppm]": mascara.get("x_max")
                            })
                df_mascaras = pd.DataFrame(filas_mascaras)
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df_mascaras.to_excel(writer, index=False, sheet_name="Mascaras_RMN1H")
                buffer.seek(0)
                if not df_mascaras.empty:
                    st.download_button("📑 Descargar máscaras RMN 1H",
                        data=buffer.getvalue(),
                        file_name=f"mascaras_{muestra['nombre']}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key=f"dl_mask_{muestra['nombre']}")

                buffer_zip = BytesIO()
                with zipfile.ZipFile(buffer_zip, "w") as zipf:
                    for e in espectros:
                        nombre_archivo = e.get("nombre_archivo", "espectro")
                        contenido = e.get("contenido")
                        if not contenido:
                            continue
                        try:
                            binario = base64.b64decode(contenido)
                            zipf.writestr(nombre_archivo, binario)
                        except:
                            continue
                    if not df_mascaras.empty:
                        zipf.writestr("mascaras_rmn1h.xlsx", buffer.getvalue())
                buffer_zip.seek(0)
                st.download_button("📦 Descargar ZIP de espectros",
                    data=buffer_zip.getvalue(),
                    file_name=f"espectros_{muestra['nombre']}.zip",
                    mime="application/zip",
                    key=f"dl_zip_{muestra['nombre']}")

    st.markdown("---")
    if st.button("Cerrar sesión"):
        st.session_state.pop("token", None)
        st.rerun()

    mostrar_sector_flotante(db)
