import numpy as np
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm, dirichlet
from copy import deepcopy

# Local files
from akula.markets import DATA_DIR

color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"
color_pink_rgb = "rgb(148, 52, 110)"
color_blue_rgb = "rgb(29,105,150)"
color_orange_rgb = "rgb(217,95,2)"
color_red_hex = "#ff2c54"
color_psi_brown = "#85543a"
color_psi_green = "#82911a"
color_psi_blue = "#003b6e"
color_psi_lblue = "#415483"
color_psi_yellow = "#fdca00"
color_psi_purple = "#7c204e"
color_psi_lpurple = "#914967"
color_psi_dgreen = "#197418"
opacity = 0.9
num_bins = 100

widths = [580, 400, 1400]
heights = [160, 160, 520]
hspacing = [0.14, 0.18, 0.04]

dirichlet_scales = {
    6555: 3.404318736838626,   # low voltage
    15739: 37.78326224978628,  # medium voltage
    8310: 21.75257995687919,   # high voltage
}

plot_lognormal = True
plot_dirichlet = True
plot_zoomed = True

dirichlet_scales = {
    6555: 3.404318736838626,   # low voltage
    15739: 37.78326224978628,  # medium voltage
    8310: 21.75257995687919,   # high voltage
}

if __name__ == "__main__":

    project = "GSA for archetypes"
    bd.projects.set_current(project)
    iterations = 20000
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    dp = bwp.load_datapackage(ZipFS(DATA_DIR / "entso-timeseries.zip"))
    tindices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]
    tdata = dp.get_resource("timeseries ENTSO electricity values.data")[0]

    ei = bd.Database("ecoinvent 3.8 cutoff")
    cols = [6555, 15739, 8310]
    # cols = [8310]

    titles_dict = {
        'electricity production, photovoltaic, 3kWp slanted-roof installation, single-Si, panel, mounted':
            "solar, single-Si",
        'electricity production, photovoltaic, 3kWp slanted-roof installation, multi-Si, panel, mounted':
            "solar, multi-Si",
        'electricity voltage transformation from medium to low voltage':
            "medium to low voltage",
        'electricity voltage transformation from high to medium voltage':
            "high to medium voltage",
        'electricity, from municipal waste incineration to generic market for electricity, medium voltage':
            "waste incineration",
        'market for electricity, high voltage':
            "high voltage",
        'electricity production, wind, >3MW turbine, onshore':
            "wind, greater 3MW",
        'heat and power co-generation, natural gas, conventional power plant, 100MW electrical':
            "natural gas, 100MW",
        'electricity production, oil':
            "oil",
        'heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014':
            "wood chips",
        'electricity production, wind, <1MW turbine, onshore':
            r"wind, lower 1MW",
        'heat and power co-generation, biogas, gas engine':
            "biogas",
        'electricity production, wind, 1-3MW turbine, offshore':
            "wind, 1-3MW",
        'heat and power co-generation, hard coal':
            "hard coal",
        'heat and power co-generation, natural gas, combined cycle power plant, 400MW electrical':
            "natural gas, 400MW",
        'electricity production, wind, 1-3MW turbine, onshore':
            "wind, 1-3MW",
        'heat and power co-generation, oil':
            "oil",
    }

    num_rows = 3
    num_cols = 7
    fig = make_subplots(
        rows=num_rows+1,
        cols=num_cols,
        vertical_spacing=0.15,
        horizontal_spacing=0.05,
        subplot_titles=["temp"]*28,
        row_heights=[0.3, 0.01,  0.3, 0.3],
    )

    showlegend = True
    row_offset = 0
    titles_str_all = []

    for i, col in enumerate(cols):
        if col in [6555, 8310]:
            col_offset = 0
        else:
            col_offset = 4

        if col in [6555, 15739]:
            irow = 1
        else:
            irow = 3

        input_rows = tindices[tindices['col'] == col]['row']
        use_rows = []
        for row in input_rows:
            ind = np.where(tindices == np.array((row, col), dtype=bwp.INDICES_DTYPE))[0][0]
            Y = tdata[ind, :iterations]
            if not np.all(Y == 0):
                use_rows.append(row)

        use_rows = sorted(use_rows)

        activity = bd.get_activity(col)
        exchanges = []
        for row in use_rows:
            for exc in activity.exchanges():
                if exc.input.id == row:
                    exchanges.append(exc)

        Y_dir = dirichlet.rvs(
            np.array([exc["amount"] for exc in exchanges]) * dirichlet_scales[col],
            size=iterations,
            )

        titles = [bd.get_activity(int(row)) for row in use_rows]
        titles_str = [f"{titles_dict.get(t['name'], t['name'])}, {t['location']}" for t in titles]
        titles_str_all += titles_str

        for j, row in enumerate(use_rows):
            if col == 8310 and j % num_cols == 0 and j > 0:
                row_offset += 1
            ind = np.where(tindices == np.array((row, col), dtype=bwp.INDICES_DTYPE))[0][0]
            Y = tdata[ind, :iterations]
            bin_min = min(Y)
            bin_max = max(Y)
            bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
            Y_samples, _ = np.histogram(Y, bins=bins_, density=True)
            midbins = (bins_[1:]+bins_[:-1])/2

            data_pos = tdata[ind, :]
            data_pos = data_pos[data_pos > 0]
            assert sum(data_pos > 1) == 0
            data_pos = data_pos[data_pos < 1]
            shape, loc, scale = lognorm.fit(data_pos, floc=0)

            Y_distr = lognorm.pdf(midbins, s=shape, scale=scale, loc=loc)

            Y_dirichlet, _ = np.histogram(Y_dir[:, j], bins=bins_, density=True)

            if plot_lognormal:
                fig.add_trace(
                    go.Scatter(
                        x=midbins,
                        y=Y_distr,
                        line=dict(color=color_red_hex),
                        name=r"$\text{Derived lognormal distributions}$",
                        showlegend=showlegend,
                        legendrank=2,
                    ),
                    row=irow + row_offset,
                    col=j % num_cols + 1 + col_offset,
                )

            if plot_dirichlet:
                fig.add_trace(
                    go.Scatter(
                        x=midbins,
                        y=Y_dirichlet,
                        name=r"$\text{Dirichlet samples}$",
                        showlegend=showlegend,
                        opacity=opacity,
                        marker=dict(opacity=opacity),
                        line=dict(color=color_psi_dgreen, width=1, shape="hvh"),
                        fill="tozeroy",
                        legendrank=3,
                    ),
                    row=irow + row_offset,
                    col=j % num_cols + 1 + col_offset,
                )

            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_samples,
                    name=r"$\text{ENTSO-E reported data}$",
                    showlegend=showlegend,
                    opacity=opacity,
                    marker=dict(opacity=opacity),
                    line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
                    fill="tozeroy",
                    legendrank=1,
                ),
                row=irow + row_offset,
                col=j % num_cols + 1 + col_offset,
            )
            showlegend = False

    if plot_zoomed:
        fig.update_yaxes(range=(-1, 12), row=1, col=1)
        fig.update_yaxes(range=(-1, 12), row=1, col=2)
        fig.update_yaxes(range=(-1, 12), row=1, col=3)

        fig.update_xaxes(range=(0.8, 1), row=1, col=5)
        fig.update_xaxes(range=(0, 0.2), row=1, col=6)

        fig.update_yaxes(range=(-0.5, 8), row=3, col=1)
        fig.update_xaxes(tickmode='array', tickvals=[0.02, 0.06], ticktext=[0.02, 0.06], row=3, col=2)
        fig.update_yaxes(range=(-2, 40), row=3, col=2)
        # fig.update_xaxes(range=(-0.001, 0.003), row=3, col=4)
        fig.update_xaxes(range=(0, 0.003), row=3, col=4)
        fig.update_yaxes(range=(-60, 1200), row=3, col=4)
        fig.update_yaxes(range=(-1, 16), row=3, col=5)
        # fig.update_xaxes(range=(-0.05, 0.4), row=3, col=6)
        fig.update_xaxes(range=(0, 0.4), row=3, col=6)
        fig.update_yaxes(range=(-0.5, 9), row=3, col=6)

        fig.update_yaxes(range=(-0.5, 11), row=4, col=1)
        # fig.update_xaxes(range=(-0.002, 0.016), row=4, col=2)
        fig.update_xaxes(range=(0, 0.016), row=4, col=2)
        fig.update_yaxes(range=(-10, 180), row=4, col=2)
        # fig.update_xaxes(range=(-0.001, 0.007), row=4, col=5)
        fig.update_xaxes(
            tickmode='array', tickvals=[0.002, 0.006], ticktext=[0.002, 0.006],
            range=(0, 0.007), row=4, col=5
        )
        fig.update_yaxes(range=(-20, 410), row=4, col=5)
        fig.update_xaxes(tickmode='array', tickvals=[0.05, 0.15], ticktext=[0.05, 0.15], row=4, col=6)
        # fig.update_xaxes(range=(-0.001, 0.015), row=4, col=7)
        fig.update_xaxes(range=(0, 0.015), row=4, col=7)
        fig.update_yaxes(range=(-20, 240), row=4, col=7)

    fig.update_xaxes(
        title_standoff=0.06,
        title_text=r"$\text{Share}$",
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_black_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )
    fig.update_yaxes(title_text=r"$\text{Frequency}$", col=1, title_standoff=0.06,)
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_black_hex,
        showline=True,
        linewidth=1,
        linecolor=color_gray_hex,
    )
    fig.update_layout(
        width=1200,
        height=550,
        legend=dict(
            yanchor="top",
            y=-0.15,  # -0.7
            xanchor="center",
            x=0.5,
            orientation='h',
            font=dict(size=13),
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=60, b=0, l=20, r=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )

    annotations = list(fig.layout.annotations)
    k = 0
    for i in range(3):
        str_ = r"$\text{" + titles_str_all[k] + "}$"
        annotations[i].update(dict(text=str_, font=dict(size=14)))
        k += 1
    annotations[3].update({'text': ""})
    for i in range(4, 6):
        str_ = r"$\text{" + titles_str_all[k] + "}$"
        annotations[i].update(dict(text=str_, font=dict(size=14)))
        k += 1
    for i in range(6, 14):
        str_ = ""
        annotations[i].update(dict(text=str_, font=dict(size=14)))
    for i in range(14, 28):
        str_ = r"$\text{" + titles_str_all[k] + "}$"
        annotations[i].update(dict(text=str_, font=dict(size=14)))
        k += 1
    fig.layout.update({"annotations": annotations})

    # Bigger titles
    fig.add_annotation(
        deepcopy(annotations[1]).update(
            dict(
                font=dict(size=16),
                text=r"$\text{Market for electricity, LOW voltage}$",
                x=annotations[1]['x'],
                y=annotations[1]['y']+0.09,
            )
        )
    )
    fig.add_annotation(
        annotations[4].update(
            dict(
                font=dict(size=16),
                text=r"$\text{Market for electricity, MEDIUM voltage}$",
                x=(annotations[4]['x'] + annotations[5]['x'])/2,
                y=annotations[4]['y']+0.09,
            )
        )
    )
    fig.add_annotation(
        annotations[17].update(
            dict(
                font=dict(size=16),
                text=r"$\text{Market for electricity, HIGH voltage}$",
                x=annotations[17]['x'],
                y=annotations[17]['y']+0.09,
            )
        )
    )
    fig.write_image(write_figs / f"_figure_4_dirichlet_validation_{plot_lognormal}_{plot_dirichlet}_{plot_zoomed}.eps")

    fig.show()

