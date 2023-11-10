import bw2data as bd
import bw2calc as bc
import bw_processing as bwp
from fs.zipfs import ZipFS
from pathlib import Path
import sys
import plotly.graph_objects as go
import numpy as np

PROJECT = "GSA with correlations"
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_DIR))


def get_one(db_name, **kwargs):
    possibles = [
        act
        for act in bd.Database(db_name)
        if all(act.get(key) == value for key, value in kwargs.items())
    ]
    if len(possibles) == 1:
        return possibles[0]
    else:
        raise ValueError(
            f"Couldn't get exactly one activity in database `{db_name}` for arguments {kwargs}"
        )


def plot(data):
    fig = go.Figure()
    fig.add_trace(go.Histogram(data))
    return fig


if __name__ == "__main__":

    bd.projects.set_current(PROJECT)
    method = ("IPCC 2013", "climate change", "GWP 100a", "uncertain")
    co = bd.Database("swiss consumption 1.0")

    dp = bwp.load_datapackage(ZipFS(str(PROJECT_DIR / "akula" / "data" / "entso-timeseries.zip")))

    data = dp.get_resource("timeseries ENTSO electricity values.data")[0]
    indices = dp.get_resource("timeseries ENTSO electricity values.indices")[0]

    # Check sums
    cols = np.unique(indices["col"])
    sums = dict()
    for col in cols:
        country_mask = indices["col"] == col
        country_data = data[country_mask, :]
        sums[col] = np.array(list(set(np.sum(country_data, axis=0))))
        if not np.allclose(sums[col], 1):
            print(col, sum(np.sum(country_data, axis=0) < 0.95))

    activity = get_one("ecoinvent 3.8 cutoff", name="market for electricity, low voltage", location="CH")
    #
    # ch_mix = indices["col"] == activity.id
    # ch_data = data[ch_mix, :]
    # ch_inds = indices[ch_mix]
    # inputs = [(bd.get_activity(row)['name'], bd.get_activity(row)['location']) for row in ch_inds["row"]]
    #
    #
    #
    # fig = plot(ch_data[:, 0])
    # fig.show()
