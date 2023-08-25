import bw2data as bd
import bw2calc as bc
import bw2io as bi
import numpy as np
from pathlib import Path
from gsa_framework.utils import read_pickle, write_pickle
from consumption_model_ch.consumption_fus import get_archetypes_scores_per_sector
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from consumption_model_ch.plot_archetypes import plot_archetypes_scores_yearly, plot_archetypes_scores_per_sector
import plotly.express as px


if __name__ == "__main__":
    project = "GSA for archetypes"
    bd.projects.set_current(project)
    co_name = "swiss consumption 1.0"
    write_dir = Path("write_files") / project.lower().replace(" ", "_") / "archetype_scores"
    write_dir.mkdir(parents=True, exist_ok=True)

    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    co = bd.Database(co_name)
    demand_act = [act for act in co if 'Food' in act['name']][0]
    lca = bc.LCA({demand_act: 1}, method)
    lca.lci()
    lca.lcia()
    print(lca.score)

    fp_scores = Path("scores.pickle")
    if fp_scores.exists():
        scores = read_pickle(fp_scores)
    else:
        scores = {}
        for exc in demand_act.exchanges():
            if exc['type'] != 'production':
                lcae = bc.LCA({exc.input: exc.amount}, method)
                lcae.lci()
                lcae.lcia()
                scores[exc.input['name']] = lcae.score
        write_pickle(scores, fp_scores)

    products_sorted = sorted(scores, key=scores.get, reverse=True)

    fig = go.Figure()
    plotly_colors = px.colors.qualitative.Antique
    colors = {product: plotly_colors[i % len(plotly_colors)] for i, product in enumerate(products_sorted)}
    num_products = 10
    for product in products_sorted[:num_products]:
        score = scores[product]
        fig.add_trace(
            go.Bar(
                name=product,
                y=[""],
                x=[score],
                orientation='h',
                marker_color=colors[product],
                showlegend=True,
            ),
        )
    color_gray_hex = "gray"
    color_black_hex = "black"
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor=color_gray_hex,
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=color_gray_hex,
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
        width=800,
        height=200,
        paper_bgcolor="rgba(255,255,255,1)",
        plot_bgcolor="rgba(255,255,255,1)",
        margin=dict(l=0, r=0, t=120, b=20),
        barmode='stack',
        legend=dict(
            orientation='h',
            traceorder='normal',
            x=0.5,
            y=1.25,
            xanchor='center',
            yanchor='bottom',
        )
    )
    fig.update_xaxes(
        title_text="LCA score, [kg CO2-eq.]"
    )
    fig.show()

    print()



# if __name__ == "__main__":
#     project = "GSA for archetypes with exiobase"
#     bd.projects.set_current(project)
#     co_name = "swiss consumption 1.0"
#     write_dir = Path("write_files") / project.lower().replace(" ", "_") / "archetype_scores"
#     write_dir.mkdir(parents=True, exist_ok=True)
#
#     method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
#     archetypes = ['archetype_Z_consumption', 'archetype_OB_consumption']
#     archetypes_scores = {}
#     for archetype in archetypes:
#         archetype_letter = archetype.replace('archetype_', "").replace('_consumption', "")
#         fp = write_dir / "monthly_{}.pickle".format(archetype)
#         archetypes_scores[archetype_letter] = read_pickle(fp)
#
#     sectors_dict = {
#         "Food": [
#             "Food and non-alcoholic beverages sector",
#             "Alcoholic beverages and tobacco sector",
#         ],
#         "Restaurants & hotels": ["Restaurants and hotels sector"],
#         "Clothing": ["Clothing and footwear sector"],
#         "Housing": ["Housing, water, electricity, gas and other fuels sector"],
#         "Furnishings": ["Furnishings, household equipment and routine household maintenance sector"],
#         "Health": ["Health sector"],
#         "Transport": ["Transport sector"],
#         "Communication": ["Communication sector"],
#         "Recreation": ["Recreation and culture sector"],
#         "Education": ["Education sector"],
#         "Other": [
#             "Durable goods sector",
#             "Fees sector",
#             "Miscellaneous goods and services sector",
#             "Other insurance premiums sector",
#             "Premiums for life insurance sector",
#         ]
#     }
#
#     num_people_dict = {
#         "A": 4.2, "B": 3.7, "C": 3.5, "D": 2.1, "E": 1.6, "F": 3.3, "G": 1.6, "H": 1.0, "I": 1.6,
#         "J": 4.2, "K": 3.2, "L": 2.4, "M": 2.2, "N": 1.2, "O": 1.1, "OA": 3.3, "OB": 1.8, "P": 2.2,
#         "Q": 1.4, "R": 1.3, "S": 2.6, "T": 2.0, "U": 1.7, "V": 2.0, "W": 1.6, "X": 2.0, "Y": 2.0, "Z": 3.3,
#     }
#     months_in_year = 12
#
#     fig = make_subplots(
#         rows=1,
#         cols=len(archetypes),
#         shared_yaxes=False,
#     )
#     plotly_colors = px.colors.qualitative.Antique
#     colors = {s: plotly_colors[i] for i, s in enumerate(sectors_dict.keys())}
#     bars = {}
#     row = 1
#
#     for sector_name, sectors in sectors_dict.items():
#         col = 1
#         showlegend = True
#         for archetype_letter, ascores in archetypes_scores.items():
#             x = months_in_year*sum([ascores[sector] for sector in sectors]) / num_people_dict[archetype_letter]
#             fig.add_trace(
#                 go.Bar(
#                     name=sector_name,
#                     y=[archetype_letter],
#                     x=[x],
#                     orientation='h',
#                     marker_color=colors[sector_name],
#                     showlegend=showlegend,
#                 ),
#                 row=row,
#                 col=col,
#             )
#             col += 1
#             showlegend = False
#
#     color_gray_hex = "gray"
#     color_black_hex = "black"
#     fig.update_xaxes(
#         showgrid=True,
#         gridwidth=1,
#         gridcolor=color_gray_hex,
#         zeroline=True,
#         zerolinewidth=1,
#         zerolinecolor=color_gray_hex,
#         showline=True,
#         linewidth=1,
#         linecolor=color_gray_hex,
#     )
#     fig.update_yaxes(
#         showgrid=True,
#         gridwidth=1,
#         gridcolor=color_gray_hex,
#         zeroline=True,
#         zerolinewidth=1,
#         zerolinecolor=color_black_hex,
#         showline=True,
#         linewidth=1,
#         linecolor=color_gray_hex,
#     )
#     fig.update_layout(
#         width=800,
#         height=200,
#         paper_bgcolor="rgba(255,255,255,1)",
#         plot_bgcolor="rgba(255,255,255,1)",
#         margin=dict(l=0, r=0, t=120, b=20),
#         barmode='stack',
#         legend=dict(
#             orientation='h',
#             traceorder='normal',
#             x=0.5,
#             y=1.25,
#             xanchor='center',
#             yanchor='bottom',
#         )
#     )
#     fig.update_xaxes(
#         title_text="LCIA score, [kg CO2-eq.]"
#     )
#     fig.show()

    print()
















