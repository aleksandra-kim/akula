import numpy as np
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm, dirichlet

# Local files
from akula.combustion import DATA_DIR

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

if __name__ == "__main__":

    project = "GSA for archetypes"
    bd.projects.set_current(project)
    iterations = 20000
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    dp = bwp.load_datapackage(ZipFS(DATA_DIR / "entso-timeseries.zip"))
    tindices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]
    tdata = dp.get_resource("timeseries ENTSO electricity values.data")[0]

    ei = bd.Database("ecoinvent 3.8 cutoff")

    showlegend = True

    for col, dirichlet_scale in dirichlet_scales.items():
        activity = bd.get_activity(col)
        rows = tindices[tindices['col'] == col]['row']
        exchanges = [exc for exc in activity.exchanges() if int(exc.input.id) in rows and exc.amount != 0]

        Y_dir = dirichlet.rvs(
            np.array([exc["amount"] for exc in exchanges]) * dirichlet_scale,
            size=iterations,
        )

        ncols = len(exchanges)
        fig = make_subplots(
            rows=1, cols=ncols,
            shared_yaxes=False,
            horizontal_spacing=0.05,
        )

        for i, exchange in enumerate(exchanges):

            ind = np.where(tindices == np.array((exchange.input.id, col), dtype=bwp.INDICES_DTYPE))[0][0]
            Y = tdata[ind, :iterations]

            bin_min = min(Y)
            bin_max = max(Y)
            bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
            midbins = (bins_[1:]+bins_[:-1])/2

            Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

            data_pos = tdata[ind, :]
            data_pos = data_pos[data_pos > 0]
            assert sum(data_pos > 1) == 0
            data_pos = data_pos[data_pos < 1]
            shape, loc, scale = lognorm.fit(data_pos, floc=0)

            Y_distr = lognorm.pdf(midbins, s=shape, scale=scale, loc=loc)

            Y_dirichlet, _ = np.histogram(Y_dir[:, i], bins=bins_, density=True)

            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_samples,
                    name=r"$\text{ENTSO electricity data}$",
                    showlegend=showlegend,
                    opacity=opacity,
                    line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
                    fill="tozeroy",
                ),
                row=1,
                col=i+1,
            )
            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_distr,
                    line=dict(color=color_red_hex),
                    name=r"$\text{Defined lognormal distributions}$",
                    showlegend=showlegend,
                ),
                row=1,
                col=i+1,
            )
            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_dirichlet,
                    name=r"$\text{Dirichlet samples}$",
                    showlegend=showlegend,
                    opacity=opacity,
                    line=dict(color=color_blue_rgb, width=1, shape="hvh"),
                    fill="tozeroy",
                ),
                row=1,
                col=i+1,
            )
            showlegend = False

            # x_title_text = r"$\text{Value, [" + exchange.input['unit'] + "]}$"
            x_title_text = r"$\text{Share}$"
            fig.update_xaxes(title_text=x_title_text, row=1, col=i+1,)

        fig.update_xaxes(
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

        fig.update_yaxes(title_text=r"$\text{Frequency}$", col=1)
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
        if col == 11989:
            space = 160
        else:
            space = 0
        fig.update_layout(
            # title=f"{exchange.input['name']}",
            width=190*ncols + space,
            height=210,
            legend=dict(
                yanchor="top",
                y=1.0,  # -0.7
                xanchor="left",
                x=1.0,
                orientation='v',
                font=dict(size=13),
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(t=30, b=0, l=30, r=0),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
        )

        fig.show()

        # fig.write_image(write_figs / f"{col}_dirichlet_electricity.pdf")
