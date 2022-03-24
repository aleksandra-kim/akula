import bw2data as bd
from bw2data.backends.schema import ActivityDataset as AD


def add_swiss_residual_mix():
    if "swiss residual electricity mix" in bd.databases:
        return

    # From notebook
    switzerland_residual = {
        "electricity production, hydro, reservoir, alpine region": 0.2814150228066876,
        "electricity production, hydro, run-of-river": 0.636056236216345,
        "heat and power co-generation, wood chips, 6667 kW, state-of-the-art 2014": 0.012048389472504549,
        "heat and power co-generation, biogas, gas engine": 0.059773867534434144,
        "heat and power co-generation, natural gas, 500kW electrical, lean burn": 0.006612375072688834,
        "electricity production, wind, >3MW turbine, onshore": 0.0010024269784498687,
        "electricity production, wind, 1-3MW turbine, onshore": 0.0026554668750543753,
        "electricity production, wind, <1MW turbine, onshore": 0.00043621504383564323,
    }

    bd.projects.set_current("GSA for archetypes")

    sr = bd.Database("swiss residual electricity mix")
    sr.register()

    # Create `ActivityDataset` as this is the only way to specify the `id`.
    AD.create(
        id=100000,
        code="CH-residual",
        database="swiss residual electricity mix",
        location="CH",
        name="swiss residual electricity mix",
        product="electricity, high voltage",
        type="process",
        data=dict(
            unit="kilowatt_hour",
            comment="Difference between generation fractions for SwissGrid and ENTSO",
            location="CH",
            name="swiss residual electricity mix",
            reference_product="electricity, high voltage",
        ),
    )

    act = bd.get_activity(id=100000)
    act.new_exchange(input=act, type="production", amount=1).save()

    act_mapping = {
        act: switzerland_residual[act["name"]]
        for act in bd.Database("ecoinvent 3.8 cutoff")
        if act["location"] == "CH"
        and act["unit"] == "kilowatt hour"
        and act["name"] in switzerland_residual
    }
    assert len(act_mapping) == len(switzerland_residual)

    for key, value in act_mapping.items():
        act.new_exchange(input=key, type="technosphere", amount=value).save()

    sr.process()

    bd.databases["ecoinvent 3.8 cutoff"]['depends'].append("swiss residual electricity mix")
    bd.databases.flush()


if __name__ == "__main__":
    add_swiss_residual_mix()
