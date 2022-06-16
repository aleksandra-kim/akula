import numpy as np
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from scipy.stats import lognorm

# Local files
from akula.combustion import DATA_DIR

plot_fig1 = True
plot_fig2 = False

if __name__ == "__main__":

    bd.projects.set_current("GSA for archetypes")
    iterations = 2000
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    dp = bwp.load_datapackage(ZipFS(DATA_DIR / "liquid-fuels-kilogram.zip"))
    tindices = dp.get_resource("liquid-fuels-tech.indices")[0]
    tdata = dp.get_resource("liquid-fuels-tech.data")[0]
    bindices = dp.get_resource("liquid-fuels-bio.indices")[0]
    bdata = dp.get_resource("liquid-fuels-bio.data")[0]

    color_gray_hex = "#b2bcc0"
    color_darkgray_hex = "#485063"
    color_black_hex = "#212931"
    color_pink_rgb = "rgb(148, 52, 110)"
    color_blue_rgb = "rgb(29,105,150)"
    color_orange_rgb = "rgb(217,95,2)"
    color_red_hex = "#ff2c54"
    opacity = 0.6
    num_bins = 100

    ind = 27
    # ind = 403
    activity = bd.get_activity(int(bindices[ind]['col']))

    if plot_fig1:

        fig = go.Figure()

        exchange = [exc for exc in activity.exchanges() if exc.input.id == bindices[ind]['row']][0]

        assert exchange['uncertainty type'] == 2
        loc = exchange['loc']
        scale = exchange['scale']
        min_distr = lognorm.ppf(0.01, s=scale, scale=np.exp(loc))
        max_distr = lognorm.ppf(0.99, s=scale, scale=np.exp(loc))

        Y = bdata[ind, :iterations]
        min_samples = min(Y)
        max_samples = max(Y)

        bin_min = min(min_distr, min_samples)
        bin_max = max(max_distr, max_samples)

        bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
        Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

        midbins = (bins_[1:]+bins_[:-1])/2
        Y_distr = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))

        showlegend = True

        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_samples,
                name=r"$\text{Balanced samples}$",
                showlegend=showlegend,
                opacity=opacity,
                line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
                fill="tozeroy",
            ),
        )
        fig.add_trace(
            go.Scatter(
                x=midbins,
                y=Y_distr,
                line=dict(color=color_red_hex),
                name=r"$\text{Defined lognormal distributions}$",
                showlegend=showlegend,
            ),
        )
        x_title_text = r"$\text{Carbon dioxide, [" + exchange['unit'] + "]}$"
        fig.update_xaxes(title_text=x_title_text)

        showlegend = False

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

        fig.update_yaxes(title_text=r"$\text{Frequency}$")
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
            # title=f"{exchange.input['name']}",
            width=280,
            height=240,
            legend=dict(
                yanchor="bottom",
                y=1.1,  # -0.7
                xanchor="center",
                x=0.5,
                orientation='h',
                font=dict(size=13),
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(t=30, b=0, l=30, r=0),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
        )

        # fig.write_image(write_figs / f"{bindices[ind]['row']}_{bindices[ind]['col']}_carbon_balance.pdf")

        fig.show()

    if plot_fig2:
        ei = bd.Database("ecoinvent 3.8 cutoff")

        opacity = 0.8
        color_psi_brown = "#85543a"
        color_psi_green = "#82911a"
        color_psi_blue = "#003b6e"
        color_psi_yellow = "#fdca00"
        color_psi_purple = "#7c204e"
        color_psi_dgreen = "#197418"
        colors = [color_darkgray_hex, color_psi_dgreen, color_psi_blue]
        symbols = ["circle", "cross", "square"]

        fig = go.Figure()

        co2 = bdata[ind, :iterations]
        rows = tindices[tindices['col'] == activity.id]['row']

        for i, row in enumerate(rows):
            name = bd.get_activity(int(row))['name']
            ind1 = np.where(tindices == np.array((row, activity.id), dtype=bwp.INDICES_DTYPE))[0][0]
            Y1 = tdata[ind1, :iterations]
            fig.add_trace(
                go.Scatter(
                    x=Y1,
                    y=co2,
                    mode="markers",
                    showlegend=True,
                    name=name,
                    opacity=opacity,
                    line=dict(color=color_darkgray_hex, width=1),
                    marker=dict(color=colors[i], symbol=symbols[i]),
                )
            )

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

        fig.update_yaxes(title_text=r"$\text{Carbon dioxide, [kg]}$")
        fig.update_xaxes(title_text=r"$\text{Fuel, [kg]}$")
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
            # title=f"{exchange.input['name']}",
            width=280,
            height=240,
            legend=dict(
                yanchor="bottom",
                y=1.1,  # -0.7
                xanchor="center",
                x=0.5,
                orientation='h',
                font=dict(size=13),
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(t=30, b=0, l=30, r=0),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
        )

        fig.write_image(write_figs / f"{bindices[ind]['col']}_carbon_balance2.pdf")

        fig.show()


    print("ss")
