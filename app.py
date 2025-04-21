
# ... (el código inicial hasta antes de la línea 228 no se muestra aquí por brevedad)

        if x and y and len(x) == len(y):
            fig, ax = plt.subplots()
            ax.scatter(x, y)
            for i, txt in enumerate(nombres):
                ax.annotate(txt, (x[i], y[i]))
            ax.set_xlabel(tipo_x)
            ax.set_ylabel(tipo_y)
            st.pyplot(fig)

            # Descargar gráfico
            buf_img = BytesIO()
            fig.savefig(buf_img, format="png")
            st.download_button("📷 Descargar gráfico", buf_img.getvalue(),
                               file_name=f"grafico_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
                               mime="image/png")
        else:
            st.warning("Los datos seleccionados no son compatibles para graficar.")
