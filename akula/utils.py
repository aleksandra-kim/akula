import bw2data as bd
import bw2calc as bc

color_gray_hex = "#b2bcc0"
color_darkgray_hex = "#485063"
color_black_hex = "#212931"


def update_fig_axes(fig):
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
        legend=dict(
            bordercolor=color_darkgray_hex,
            borderwidth=1,
        ),
        margin=dict(t=40, b=10, l=10, r=0),
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
    )
    return fig


def setup_bw_project(years="151617"):
    project = "GSA for archetypes"
    bd.projects.set_current(project)

    co = bd.Database('swiss consumption 1.0')
    fu = [act for act in co if f"ch hh average consumption aggregated, years {years}" == act['name']][0]
    demand = {fu: 1}
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")

    lca = bc.LCA(demand=demand, method=method, use_distributions=False)
    lca.lci()
    lca.lcia()

    return lca
