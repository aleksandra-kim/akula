import bw2data as bd
import bw_processing as bwp
from pathlib import Path
from fs.zipfs import ZipFS
from copy import deepcopy

DATA_DIR = Path(__file__).parent.parent.resolve() / "data"


def replace_ei_with_entso():
    """Replace ecoinvent electricity mixes with ENTSO data averaged over the years 2019-2021."""

    bd.projects.set_current("GSA for archetypes")

    fp_entso = DATA_DIR / "entso-average.zip"
    dp = bwp.load_datapackage(ZipFS(str(fp_entso)))

    indices = dp.get_resource('average ENTSO electricity values.indices')[0]
    data = dp.get_resource('average ENTSO electricity values.data')[0]

    for i, inds in enumerate(indices):

        if i % 100 == 0:
            print(f"{i}/{len(indices)} exchanges processed")

        activity = bd.get_activity(int(inds['col']))
        exchange = [exc for exc in activity.exchanges() if exc.input.id == int(inds['row'])]

        if len(exchange) == 1:
            exchange = exchange[0]
            exchange['original_amount'] = deepcopy(exchange['amount'])
            exchange['amount'] = data[i]
            exchange['loc'] = data[i]
            exchange.save()
            activity.save()

        elif len(exchange) == 0:
            if inds['row'] != inds['col']:
                type_ = "technosphere"
            else:
                type_ = "production"
            input_activity = bd.get_activity(int(inds['row']))
            activity.new_exchange(
                input=(input_activity['database'], input_activity["code"]),
                amount=data[i],
                type=type_,
                original_amount=0,
            ).save()

        else:
            print(exchange)


if __name__ == "__main__":
    replace_ei_with_entso()
