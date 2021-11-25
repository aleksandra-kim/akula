import bw2calc as bc
import bw2data as bd
from pathlib import Path
import numpy as np

from consumption_model_ch.consumption_fus import (
    add_archetypes_consumption,
    get_archetypes_scores_per_month,
    get_archetypes_scores_per_sector,
)

from consumption_model_ch.plot_archetypes import plot_archetypes_scores_yearly, plot_archetypes_scores_per_sector

use_exiobase = False
add_archetypes = False


if __name__ == "__main__":

    path_base = Path('/Users/akim/Documents/LCA_files/')

    co_name = "swiss consumption 1.0"
    co = bd.Database(co_name)
    if use_exiobase:
        project = "GSA for archetypes with exiobase"
    else:
        project = "GSA for archetypes"
    bd.projects.set_current(project)
    write_dir = Path("write_files") / project.lower().replace(" ", "_") / "archetype_scores"
    write_dir.mkdir(parents=True, exist_ok=True)

    # Add archetypes and compute total yearly scores per archetype
    if add_archetypes:
        add_archetypes_consumption(co_name)
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    fp_archetypes_scores = write_dir / "monthly_scores.pickle"
    archetypes_scores_monthly = get_archetypes_scores_per_month(co_name, method, fp_archetypes_scores)
    # Compare with Andi's results (reproduce Fig. 3 in Andi's data mining paper, top part)
    num_months_in_year = 12
    archetypes_scores_yearly = {
        archetype: score*num_months_in_year for archetype, score in archetypes_scores_monthly.items()
    }
    fig = plot_archetypes_scores_yearly(archetypes_scores_yearly)
    # fig.write_html(write_dir / "yearly_scores.html")
    fig.write_image(write_dir / "yearly_scores.pdf")
    fig.show()
    # Compare with Andi's contributions per sectors (reproduce Fig. 3 in Andi's data mining paper, bottom part)
    fp_archetypes_scores_sectors = write_dir / "monthly_scores.pickle"
    archetypes_scores_per_sector = get_archetypes_scores_per_sector(co_name, method, write_dir)
    fig = plot_archetypes_scores_per_sector(archetypes_scores_per_sector)
    # fig.write_html(write_dir / "sector_scores.html")
    fig.write_image(write_dir / "sector_scores.pdf")
    fig.show()

    # hh_average = [act for act in co if "ch hh average consumption aggregated" == act['name']][0]
    print()
