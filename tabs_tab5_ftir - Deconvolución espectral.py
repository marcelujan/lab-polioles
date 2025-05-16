    # --- Deconvolución espectral (estilo minimalista, sin inputs adicionales) ---
    st.subheader("🔍 Deconvolución FTIR")
    if st.checkbox("Activar deconvolución", key="activar_deconv") and not df_espectros.empty:
        opciones = df_espectros.apply(lambda row: f"{row['muestra']} – {row['tipo']} – {row['archivo']}", axis=1).tolist()
        espectro_sel = st.selectbox("Seleccionar espectro", opciones)
        fila = df_espectros.iloc[opciones.index(espectro_sel)]

        try:
            import numpy as np
            from scipy.optimize import curve_fit
            from sklearn.metrics import mean_squared_error, r2_score

            contenido = BytesIO(base64.b64decode(fila["contenido"]))
            ext = fila["archivo"].split(".")[-1].lower()
            if ext == "xlsx":
                df = pd.read_excel(contenido)
            else:
                for sep in [",", ";", "\t", " "]:
                    contenido.seek(0)
                    try:
                        df = pd.read_csv(contenido, sep=sep)
                        if df.shape[1] >= 2:
                            break
                    except:
                        continue
                else:
                    df = None

            if df is not None:
                df = df.iloc[:, :2]
                df.columns = ["x", "y"]
                df = df.apply(pd.to_numeric, errors="coerce").dropna()

                # Usar rango global ya definido por el usuario si existe
                x_min = st.session_state.get("x_min", float(df["x"].min()))
                x_max = st.session_state.get("x_max", float(df["x"].max()))
                df_fit = df[(df["x"] >= x_min) & (df["x"] <= x_max)].sort_values(by="x")

                def multi_gaussian(x, *params):
                    y = np.zeros_like(x)
                    for i in range(0, len(params), 3):
                        amp, cen, wid = params[i:i+3]
                        y += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
                    return y

                n_gauss = st.slider("Cantidad de gaussianas", 1, 6, 2)
                p0 = []
                for i in range(n_gauss):
                    p0 += [df_fit["y"].max()/n_gauss, df_fit["x"].min() + i * (df_fit["x"].ptp() / n_gauss), 10]

                try:
                    popt, _ = curve_fit(multi_gaussian, df_fit["x"], df_fit["y"], p0=p0)
                    y_fit = multi_gaussian(df_fit["x"], *popt)

                    fig, ax = plt.subplots()
                    ax.plot(df_fit["x"], df_fit["y"], label="Original")
                    ax.plot(df_fit["x"], y_fit, "--", label="Ajuste")

                    resultados = []
                    for i in range(n_gauss):
                        amp, cen, wid = popt[3*i:3*i+3]
                        gauss = amp * np.exp(-(df_fit["x"] - cen)**2 / (2 * wid**2))
                        area = amp * wid * np.sqrt(2*np.pi)
                        ax.plot(df_fit["x"], gauss, ":", label=f"Pico {i+1}")
                        resultados.append({
                            "Pico": i+1,
                            "Centro (cm⁻¹)": round(cen, 2),
                            "Amplitud": round(amp, 2),
                            "Anchura σ": round(wid, 2),
                            "Área": round(area, 2)
                        })

                    ax.legend()
                    st.pyplot(fig)

                    rmse = mean_squared_error(df_fit["y"], y_fit, squared=False)
                    r2 = r2_score(df_fit["y"], y_fit)
                    st.markdown(f"**RMSE:** {rmse:.4f} &nbsp;&nbsp; **R²:** {r2:.4f}")

                    df_result = pd.DataFrame(resultados)
                    st.dataframe(df_result, use_container_width=True)

                    buf_excel = BytesIO()
                    with pd.ExcelWriter(buf_excel, engine="xlsxwriter") as writer:
                        df_result.to_excel(writer, index=False, sheet_name="Deconvolucion")
                    buf_excel.seek(0)
                    st.download_button("📥 Descargar parámetros", data=buf_excel.getvalue(),
                                       file_name="deconvolucion_resultados.xlsx",
                                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                except:
                    st.warning("No se pudo ajustar el modelo. Ajustá el número de picos.")
        except Exception as e:
            st.error(f"Error al procesar espectro: {e}")



    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    nombre_base = f"FTIR_{now}"

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="xlsxwriter") as writer:
                resumen.to_excel(writer, index=False, sheet_name="Resumen")
                for muestra, tipo, archivo, df in datos:
                    df_filtrado = df[(df.iloc[:, 0] >= x_min) & (df.iloc[:, 0] <= x_max)]
                    df_filtrado.to_excel(writer, index=False, sheet_name=f"{muestra[:15]}_{tipo[:10]}")
                if fwhm_rows:
                    df_fwhm = pd.DataFrame(fwhm_rows)
                    df_fwhm = df_fwhm.sort_values(by="Muestra")
                    df_fwhm.to_excel(writer, index=False, sheet_name="Picos_FWHM")
    buffer_excel.seek(0)
    st.download_button("📥 Descargar Excel", data=buffer_excel.getvalue(), file_name=f"{nombre_base}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    buffer_img = BytesIO()
    fig.savefig(buffer_img, format="png", dpi=300, bbox_inches="tight")
    st.download_button("📷 Descargar PNG", data=buffer_img.getvalue(), file_name=f"{nombre_base}.png", mime="image/png")
