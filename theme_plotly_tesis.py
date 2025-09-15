# theme_plotly_tesis.py
import plotly.graph_objs as go
import plotly.io as pio

FONT_FAMILY = "Times New Roman"  # cámbialo si tu reglamento pide otro

tesis_template = go.layout.Template(
    layout=go.Layout(
        font=dict(family=FONT_FAMILY, color="black", size=16),
        title=dict(font=dict(family=FONT_FAMILY, color="black", size=18)),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=["#111111", "#404040", "#6b6b6b", "#9e9e9e", "#c7c7c7"],
        margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=14),
                    orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        xaxis=dict(
            showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
            zeroline=False, showline=True, linecolor="black", linewidth=1.5,
            ticks="outside", tickcolor="black", ticklen=6, tickwidth=1.5,
            titlefont=dict(color="black"), tickfont=dict(color="black")
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e0e0e0", gridwidth=1,
            zeroline=False, showline=True, linecolor="black", linewidth=1.5,
            ticks="outside", tickcolor="black", ticklen=6, tickwidth=1.5,
            titlefont=dict(color="black"), tickfont=dict(color="black")
        ),
    ),
    data={
        "scatter": [go.Scatter(mode="lines", line=dict(width=2), marker=dict(size=6))],
        "bar": [go.Bar(marker=dict(line=dict(color="black", width=1)))],
    }
)

def setup_plotly_tesis(default=True):
    pio.templates["tesis"] = tesis_template
    if default:
        pio.templates.default = "tesis"

def add_tesis_guides(fig, y0=True, x_lines=None, dash="dash", width=2):
    # línea horizontal en y=0
    if y0:
        fig.add_shape(type="line", xref="paper", yref="y",
                      x0=0, x1=1, y0=0, y1=0,
                      line=dict(color="black", dash=dash, width=width), layer="above")
    # líneas verticales (p.ej. [1440])
    if x_lines:
        for xv in x_lines:
            fig.add_shape(type="line", xref="x", yref="paper",
                          x0=xv, x1=xv, y0=0, y1=1,
                          line=dict(color="black", dash=dash, width=width), layer="above")
