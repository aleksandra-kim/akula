import numpy as np
import plotly.graph_objects as go
from pathlib import Path


np.random.seed(2344)
samples = 9000

s = 2
m = 9
Y1 = s/5 * np.random.randn(samples) + m/2 + np.random.lognormal(np.log(0.55*m), s/5, samples)
Y2 = s/11 * np.random.randn(samples) + m/2.1 + np.random.lognormal(np.log(0.55*m), s/11, samples)
print(np.mean(Y1), np.mean(Y2))

bin_min, bin_max = 5, 19
num_bins = 100
opacity = 1.0

include_reduced = True
include_average = True

bins_ = np.linspace(bin_min, bin_max, num_bins, endpoint=True)

color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"
lca_scores_axis_title = r"$\textbf{Carbon footprint per capita, [tCO}_2\textbf{-eq}]$"
# color_bg_fg = "#16191e"
color_bg_fg = "black"
# color_bg = "rgb(148, 52, 110)"
color_bg = "#0147ab"
# color_bg_fg = "rgb(29,105,150)"
# color_bg = "rgb(148, 52, 110)"
color_marker = "#d21404"

fig = go.Figure()

# Complete uncertainty
freq1, bins1 = np.histogram(Y1, bins=bins_)
fig.add_trace(
    go.Scatter(
        x=bins1,
        y=freq1,
        opacity=opacity,
        line=dict(color=color_bg_fg, width=2, shape="hvh"),
        name=r"$\text{Overall uncertainty}$",
        showlegend=True,
        fill="tozeroy",
    ),
)

# Reduced uncertainty
if include_reduced:
    freq2, bins2 = np.histogram(Y2, bins=bins_)
    fig.add_trace(
        go.Scatter(
            x=bins2,
            y=freq2,
            opacity=opacity,
            line=dict(color=color_bg, width=2, shape="hvh"),
            name=r"$\text{Reduced uncertainty}$",
            showlegend=True,
            fill="tozeroy",
        ),
    )

if include_average:
    fig.add_trace(
        go.Scatter(
            x=[9.4],
            y=[0],
            name=r"$\text{Average carbon footprint}$",
            showlegend=True,
            mode="markers",
            marker=dict(size=20, symbol="x", color=color_marker),
        ),
    )

fig.update_xaxes(
    title_text=lca_scores_axis_title,
    title_standoff=12,
    title_font_size=20,
    title_font_color="black",
    tickfont_size=16,
    tickfont_color="black",
    showgrid=True,
    gridwidth=2,
    gridcolor=color_gray_hex,
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor=color_gray_hex,
    showline=True,
    linewidth=2,
    linecolor=color_gray_hex,
)
fig.update_yaxes(
    title_text=r"$\textbf{Frequency}$",
    title_standoff=6,
    title_font_size=20,
    title_font_color="black",
    tickfont_size=16,
    tickfont_color="black",
    range=[-40, 640],
    showgrid=True,
    gridwidth=2,
    gridcolor=color_gray_hex,
    zeroline=True,
    zerolinewidth=2,
    zerolinecolor=color_black_hex,
    showline=True,
    linewidth=2,
    linecolor=color_gray_hex,
)
fig.update_layout(
    width=540,
    height=220,
    # paper_bgcolor="rgba(255,255,255,1)",
    # plot_bgcolor="rgba(255,255,255,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(
        x=0.7,
        y=0.90,
        orientation="v",
        xanchor="center",
        font=dict(size=18, color="black"),
        # bgcolor=color_lightgray_hex,
        bordercolor=color_darkgray_hex,
        borderwidth=0,
        bgcolor="rgba(255,255,255,0.6)",
    ),
    margin=dict(l=65, r=0, t=0, b=60),
)

fig.show()

if not include_reduced and not include_average:
    filepath_fig = Path(f"lca_scores_base.png")
elif include_reduced and include_average:
    filepath_fig = Path(f"lca_scores_base_reduced_average.png")
elif not include_reduced and include_average:
    filepath_fig = Path(f"lca_scores_base_average.png")

fig.write_image(filepath_fig.as_posix())
