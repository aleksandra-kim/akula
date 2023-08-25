import numpy as np
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm, dirichlet

# Local files
from akula.electricity.create_datapackages import DATA_DIR

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
color_psi_yellow = "#fdca00"
color_psi_purple = "#7c204e"
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

plot_lognormal = False
plot_dirichlet = False
plot_zoomed = False

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
    # dk = [act for act in ei if 'market for electricity' in act['name'] and 'DK' in act['location']]
    cols = [6555, 15739, 8310]
    cols = [8310]

    titles_dict = {
        'electricity production, photovoltaic, 3kWp slanted-roof installation, single-Si, panel, mounted':
            "photovoltaic, single-Si",
        'electricity production, photovoltaic, 3kWp slanted-roof installation, multi-Si, panel, mounted':
            "photovoltaic, multi-Si",
        'electricity voltage transformation from medium to low voltage':
            "medium to low voltage",
        'electricity voltage transformation from high to medium voltage':
            "high to medium voltage",
        'electricity, from municipal waste incineration to generic market for electricity, medium voltage':
            "waste incineration",
        'market for electricity, high voltage':
            "high voltage",
        'electricity production, wind, >3MW turbine, onshore':
            "wind, >3MW",
        'heat and power co-generation, natural gas, conventional power plant, 100MW electrical':
            "natural gas, 100MW",
        'electricity production, oil':
            "oil",
        'heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014':
            "wood chips",
        'electricity production, wind, <1MW turbine, onshore':
            "wind, <1MW",
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

    num_cols = 7
    showlegend = True
    for i, col in enumerate(cols):
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

        fig = make_subplots(
            rows=len(use_rows) // num_cols + 1, cols=min(num_cols, len(use_rows)),
            subplot_titles=titles_str,
            vertical_spacing=0.16,
            horizontal_spacing=hspacing[i],
        )

        k = 0
        for j, row in enumerate(use_rows):
            if j % num_cols == 0:
                k += 1
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
                        name=r"$\text{Defined lognormal distributions}$",
                        showlegend=showlegend,
                        legendrank=2,
                    ),
                    row=k,
                    col=j % num_cols + 1,
                )

            if plot_dirichlet:
                fig.add_trace(
                    go.Scatter(
                        x=midbins,
                        y=Y_dirichlet,
                        name=r"$\text{Dirichlet samples}$",
                        showlegend=showlegend,
                        opacity=opacity,
                        line=dict(color=color_psi_dgreen, width=1, shape="hvh"),
                        fill="tozeroy",
                        legendrank=3,
                    ),
                    row=k,
                    col=j % num_cols + 1,
                )

            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_samples,
                    name=r"$\text{ENTSO-E reported data}$",
                    showlegend=showlegend,
                    opacity=opacity,
                    line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
                    fill="tozeroy",
                    legendrank=1,
                ),
                row=k,
                col=j % num_cols + 1,
            )
            showlegend = False
            if i == 2:
                if j == 1 or j == 13:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=[0.02, 0.06],
                        ticktext=[0.02, 0.06],
                        row=k,
                        col=j % num_cols + 1,
                    )
                if j == 3:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=[0.005, 0.015],
                        ticktext=[0.005, 0.015],
                        row=k,
                        col=j % num_cols + 1,
                    )
                if j == 11:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=[0.002, 0.006],
                        ticktext=[0.002, 0.006],
                        row=k,
                        col=j % num_cols + 1,
                    )
                if j == 12:
                    fig.update_xaxes(
                        tickmode='array',
                        tickvals=[0.05, 0.15],
                        ticktext=[0.05, 0.15],
                        row=k,
                        col=j % num_cols + 1,
                    )
        if plot_zoomed:
            if i == 0:
                fig.update_yaxes(range=(-1, 12), row=1)
            elif i == 1:
                fig.update_xaxes(range=(0.8, 1), row=1, col=1)
                fig.update_xaxes(range=(0, 0.2), row=1, col=2)
            elif i == 2:
                fig.update_yaxes(range=(-0.5, 8), row=1, col=1)
                fig.update_yaxes(range=(-2, 40), row=1, col=2)
                # fig.update_xaxes(range=(-0.001, 0.003), row=1, col=4)
                fig.update_xaxes(range=(0, 0.003), row=1, col=4)
                fig.update_yaxes(range=(-60, 1200), row=1, col=4)
                fig.update_yaxes(range=(-1, 16), row=1, col=5)
                # fig.update_xaxes(range=(-0.05, 0.4), row=1, col=6)
                fig.update_xaxes(range=(0, 0.4), row=1, col=6)
                fig.update_yaxes(range=(-0.5, 9), row=1, col=6)

                fig.update_yaxes(range=(-0.5, 11), row=2, col=1)
                # fig.update_xaxes(range=(-0.002, 0.016), row=2, col=2)
                fig.update_xaxes(range=(0, 0.016), row=2, col=2)
                fig.update_yaxes(range=(-10, 180), row=2, col=2)
                # fig.update_xaxes(range=(-0.001, 0.007), row=2, col=5)
                fig.update_xaxes(range=(0, 0.007), row=2, col=5)
                fig.update_yaxes(range=(-20, 410), row=2, col=5)
                # fig.update_xaxes(range=(-0.001, 0.015), row=2, col=7)
                fig.update_xaxes(range=(0, 0.015), row=2, col=7)
                fig.update_yaxes(range=(-20, 240), row=2, col=7)

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
            width=widths[i],
            height=600,#heights[i],
            legend=dict(
                yanchor="bottom",
                y=1.1,  # -0.7
                xanchor="left",
                x=0.0,
                orientation='v',
                font=dict(size=13),
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(t=40, b=10, l=10, r=40),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
        )

        fig.write_image(write_figs / f"{col}_electricity_market_{plot_lognormal}_{plot_dirichlet}_{plot_zoomed}.pdf")

        fig.show()

        # print("")
