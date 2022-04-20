import numpy as np
import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import lognorm

# Local files
from akula.parameterized_exchanges import DATA_DIR, get_parameters, get_lookup_cache


def get_samples(exc):
    inds = np.array(
        (
            lookup_cache[exc['input']],
            lookup_cache[(activity['database'], activity['code'])]
        ),
        dtype=bwp.INDICES_DTYPE
    )
    if exc['type'] == "biosphere":
        mask = bindices == inds
        x = bdata[mask, :].flatten()
    else:
        mask = tindices == inds
        x = tdata[mask, :].flatten()
    x = np.random.choice(x, iterations)
    if exc['amount'] < 0:
        x = -x
    return x


if __name__ == "__main__":

    bd.projects.set_current("GSA for archetypes")
    iterations = 2000
    write_figs = Path("/Users/akim/PycharmProjects/akula/dev/write_files/paper3")

    lookup_cache = get_lookup_cache()

    dp = bwp.load_datapackage(ZipFS(DATA_DIR / "ecoinvent-parameterization.zip"))
    tindices = dp.get_resource("ecoinvent-parameterization-tech.indices")[0]
    tdata = dp.get_resource("ecoinvent-parameterization-tech.data")[0]
    tflip = dp.get_resource("ecoinvent-parameterization-tech.flip")[0]
    bindices = dp.get_resource("ecoinvent-parameterization-bio.indices")[0]
    bdata = dp.get_resource("ecoinvent-parameterization-bio.data")[0]

    parameters = get_parameters()

    itrain = 78
    imaize_RoW = 224
    imaize_ZA = 239
    icar = 424
    ihydro = 452
    isilage = 484

    count = 0
    for p, parameter in enumerate(parameters[6:7]):
        activity = parameter['activity']
        id_ = lookup_cache[(activity['database'], activity['code'])]
        iexcs = [int(el.replace("__exchange_", ""))for el in list(parameter['exchanges'])]
        exchanges = [activity['exchanges'][i] for i in iexcs if activity['exchanges'][i].get('uncertainty type', 0) > 1]

        if len(exchanges) < 0:
            continue

        # Make figure
        #############

        color_gray_hex = "#b2bcc0"
        color_darkgray_hex = "#485063"
        color_black_hex = "#212931"
        color_pink_rgb = "rgb(148, 52, 110)"
        color_blue_rgb = "rgb(29,105,150)"
        color_orange_rgb = "rgb(217,95,2)"
        color_red_hex = "#ff2c54"
        opacity = 0.6
        num_bins = 100

        cols = min(len(exchanges), 6)
        rows = int(np.ceil(len(exchanges) / cols))

        exc_names = [exc['name'].replace("passenger car", "").rstrip(", ").lstrip(", ") for exc in exchanges]
        exc_names = [r"$\text{" + r"{}".format(exc[:40]) + "}$" for exc in exc_names]

        x_title_text = [r"$\text{Value, [" + exc['unit'] + "]}$" for exc in exchanges]

        fig = make_subplots(
            rows=rows, cols=cols,
            horizontal_spacing=0.08,
            vertical_spacing=0.06,
            shared_yaxes=False,
            subplot_titles=exc_names,
        )

        showlegend = True

        for i, exchange in enumerate(exchanges):

            assert exchange['uncertainty type'] == 2
            loc = exchange['loc']
            scale = exchange['scale']
            min_distr = lognorm.ppf(0.01, s=scale, scale=np.exp(loc))
            max_distr = lognorm.ppf(0.99, s=scale, scale=np.exp(loc))

            Y = get_samples(exchange)
            min_samples = min(Y)
            max_samples = max(Y)

            bin_min = min(min_distr, min_samples)
            bin_max = max(max_distr, max_samples)

            bins_ = np.linspace(bin_min, bin_max, num_bins+1, endpoint=True)
            Y_samples, _ = np.histogram(Y, bins=bins_, density=True)

            midbins = (bins_[1:]+bins_[:-1])/2
            Y_distr = lognorm.pdf(midbins, s=scale, scale=np.exp(loc))

            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_samples,
                    name=r"$\text{Parameterized samples}$",
                    showlegend=showlegend,
                    opacity=opacity,
                    line=dict(color=color_darkgray_hex, width=1, shape="hvh"),
                    fill="tozeroy",
                ),
                row=i // cols + 1,
                col=i % cols + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=midbins,
                    y=Y_distr,
                    line=dict(color=color_red_hex),
                    name=r"$\text{Defined lognormal distributions}$",
                    showlegend=showlegend,
                ),
                row=i // cols + 1,
                col=i % cols + 1,
            )
            fig.update_xaxes(title_text=x_title_text[i], col=i+1)

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

        # fig.update_xaxes(
        #     tickmode='array',
        #     ticktext=[0.0007, 0.001, 0.0013],
        #     tickvals=[0.0007, 0.001, 0.0013],
        #     row=1,
        #     col=3,
        # )

        # xpos = [0.08, 0.29, 0.5, 0.71, 0.92]
        # for i in range(cols):
        #     fig.add_annotation(
        #         {
        #             'font': {'size': 14},
        #             'showarrow': False,
        #             'text': names1[i],
        #             'x': xpos[i]-0.08,
        #             'xanchor': 'left',
        #             'xref': 'paper',
        #             'y': 1.2,
        #             'yanchor': 'bottom',
        #             'yref': 'paper'
        #         }
        #     )
        #     fig.add_annotation(
        #         {
        #             'font': {'size': 14},
        #             'showarrow': False,
        #             'text': names2[i],
        #             'x': xpos[i]-0.08,
        #             'xanchor': 'left',
        #             'xref': 'paper',
        #             'y': 1.05,
        #             'yanchor': 'bottom',
        #             'yref': 'paper'
        #         }
        #     )

        fig.add_annotation(
            {
                'font': {'size': 14},
                'showarrow': False,
                'text': activity['name'].upper() + ", " + activity['location'].upper(),
                'x': 0,
                'xanchor': 'left',
                'xref': 'paper',
                'y': 1.08,
                'yanchor': 'bottom',
                'yref': 'paper'
            }
        )

        fig.update_layout(
            # width=220*cols,
            width=260*cols,
            height=300*rows,
            legend=dict(
                yanchor="bottom",
                y=1.04,  # -0.7
                xanchor="center",
                x=0.5,
                orientation='h',
                font=dict(size=13),
                bordercolor=color_darkgray_hex,
                borderwidth=1,
            ),
            margin=dict(t=100, b=10, l=10, r=0),
            paper_bgcolor="rgba(255,255,255,1)",
            plot_bgcolor="rgba(255,255,255,1)",
        )

        fig.write_image(write_figs / f"{p}_id{id_}_{activity['name'][:40]}_{activity['location'][:3]}.pdf")

        fig.show()
