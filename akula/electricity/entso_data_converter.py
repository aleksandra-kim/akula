import bw2data as bd
import pandas as pd
import numpy as np
from collections import defaultdict
# import bentso
from bentso import CachingDataClient as CDC
from bentso.constants import ENTSO_COUNTRIES, TRADE_PAIRS
from pathlib import Path
from config import ENTSO_MAPPING

assert bd.__version__ >= (4, 0, "DEV11")
# assert bentso.__version__ >= (0, 4)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
cdc = CDC()


"""
Get electricity data from the ENTSO-E API

To run this script, you must have the following environment variables set:

    BENTSO_DATA_DIR: Directory to cache data from ENTSO-E API
    ENTSOE_API_TOKEN: API token you get from signing up to ENTSO-E transparency platform

Request an API key by sending an email to transparency@entsoe.eu with “Restful API access” in the subject line. 
In the email body state your registered email address. You will receive an email when you have been provided 
with the API key. The key is then visible in your ENTSO-E account under “Web API Security Token”. 
Source: https://thesmartinsights.com/how-to-query-data-from-the-entso-e-transparency-platform-using-python/ 

"""


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


def is_generation_exchange(exc):
    return (
        exc.input["unit"] == "kilowatt hour"
        and "import from" not in exc.input["name"]
        and not exc.input["name"].startswith("market")
        and "voltage transformation" not in exc.input["name"]
    )


def is_trade_exchange(exc):
    return exc.input["unit"] == "kilowatt hour" and "import from" in exc.input["name"]


def apply_country_specific_fixes(df, country):
    if country in ("GR", "ME"):
        df.rename(
            columns={"Hydro Water Reservoir": "Hydro Run-of-river and poundage"},
            inplace=True,
        )
    elif country == "NL":
        df.drop("Waste", axis=1, inplace=True)
    elif country == "EE":
        df.rename(columns={"Fossil Oil shale": "Fossil Oil"}, inplace=True)
    elif country == "NO":
        df["Hydro Water Reservoir"] += df["Hydro Run-of-river and poundage"]
        df.drop("Hydro Run-of-river and poundage", axis=1, inplace=True)
    elif country in ("RO", "BG"):
        df["Hydro Run-of-river and poundage"] += df["Hydro Water Reservoir"]
        df.drop("Hydro Water Reservoir", axis=1, inplace=True)
    elif country == "BA":
        # Very wrong
        df["Fossil Brown coal/Lignite"] += df["Fossil Hard coal"]
        df.drop("Fossil Hard coal", axis=1, inplace=True)
    return df


def get_df(country, year):
    df = cdc.get_generation(
        country=country, year=year, clean=True, full_year=True, fix_lv=True
    )
    df = apply_country_specific_fixes(df, country)
    return df


def get_swissgrid_df(fp):
    df = pd.read_excel(fp, sheet_name="Zeitreihen0h15", usecols=[2], header=1)
    df = df.groupby(df.index // 4).sum()
    return df


class ENTSODataConverter:
    def __init__(self, ecoinvent_database):
        self.ei = bd.Database(ecoinvent_database)

        # Prepare cache of activities
        self.hv_mixes = {
            act["location"]: act
            for act in self.ei
            if act["name"] == "market for electricity, high voltage"
        }
        self.mv_mixes = {
            act["location"]: act
            for act in self.ei
            if act["name"] == "market for electricity, medium voltage"
            and act["location"] in ENTSO_COUNTRIES
        }
        self.lv_mixes = {
            act["location"]: act
            for act in self.ei
            if act["name"] == "market for electricity, low voltage"
            and act["location"] in ENTSO_COUNTRIES
        }
        self.mv_loss_coefficients = {
            act["location"]: next(iter(act.technosphere()))["amount"]
            for act in self.ei
            if act["name"]
            == "electricity voltage transformation from high to medium voltage"
            and act["location"] in ENTSO_COUNTRIES
        }
        self.lv_loss_coefficients = {
            act["location"]: next(iter(act.technosphere()))["amount"]
            for act in self.ei
            if act["name"]
            == "electricity voltage transformation from medium to low voltage"
            and act["location"] in ENTSO_COUNTRIES
        }

        self.swissgrid_total = (
            pd.concat(
                [
                    get_swissgrid_df(fp)
                    for fp in (
                        DATA_DIR / "EnergieUebersichtCH-2019.xls",
                        DATA_DIR / "EnergieUebersichtCH-2020.xlsx",
                        DATA_DIR / "EnergieUebersichtCH-2021.xlsx",
                    )
                ]
            )["kWh.1"]
            / 1000
        )  # Convert to MWh

    def get_ratio_hv_lv_solar_ecoinvent(self, country):
        """Get the fraction of high voltage solar generation in total solar generation as defined in ecoinvent markets"""
        hv_solar_in_low_voltage = (
            sum(
                exc["amount"]
                for exc in self.hv_mixes[country].technosphere()
                if exc.input["name"] in ENTSO_MAPPING["Solar"]
            )
            / self.mv_loss_coefficients[country]
            / self.lv_loss_coefficients[country]
            * next(
                exc["amount"]
                for exc in self.lv_mixes[country].technosphere()
                if exc.input["name"]
                == "electricity voltage transformation from medium to low voltage"
            )
        )
        lv_solar_in_low_voltage = sum(
            exc["amount"]
            for exc in self.lv_mixes[country].technosphere()
            if exc.input["name"] in ENTSO_MAPPING["Solar"]
        )
        denominator = (hv_solar_in_low_voltage + lv_solar_in_low_voltage) or 1
        return hv_solar_in_low_voltage / denominator

    def adjust_entso_df_for_low_voltage(self, df, country):
        """Adjust ENTSO dataframe for use in ecoinvent high voltage markets.

        1. Remove medium voltage waste treatment category.
        2. Adjust solar generation to only include high voltage generation.
        3. Convert all ENTSO categories to ecoinvent generators (i.e. disaggregate), using ecoinvent ratios.

        """
        market = self.lv_mixes[country]

        if "Solar" not in df.columns:
            return (None, pd.Series(np.ones(df.shape[0]), index=df.index))

        solar_lv = df["Solar"] * (1 - self.get_ratio_hv_lv_solar_ecoinvent(country))

        # This is now only high voltage
        df["Solar"] *= self.get_ratio_hv_lv_solar_ecoinvent(country)

        if "Waste" in df.columns:
            prelim_gen = (df.sum(axis=1) - df["Waste"])
            waste_gen = (df["Waste"] / self.lv_loss_coefficients[country])
        else:
            prelim_gen = df.sum(axis=1)
            waste_gen = 0

        mv_generation_with_losses = (
            prelim_gen
            / self.mv_loss_coefficients[country]
            / self.lv_loss_coefficients[country]
        ) + waste_gen

        amounts = {
            exc.input["name"]: exc["amount"]
            for exc in market.technosphere()
            if is_generation_exchange(exc) and exc["amount"]
        }
        if not amounts:
            print(
                "Warning with {}, average low voltage generation of {} but no ecoinvent exchanges".format(
                    country, solar_lv.mean(axis=0)
                )
            )
            return (None, pd.Series(np.ones(df.shape[0]), index=df.index))

        subtotal = sum(amounts.values())
        disaggregated = pd.DataFrame(
            {name: amount / subtotal * solar_lv for name, amount in amounts.items()},
            index=df.index,
        ).fillna(0)
        return disaggregated, mv_generation_with_losses

    def data_dict_for_low_voltage_market(
        self, country, years=(2019, 2020, 2021), average=False
    ):
        data = {}

        market = self.lv_mixes[country]
        df, series = self.adjust_entso_df_for_low_voltage(
            pd.concat([get_df(country, year) for year in years]), country
        )

        imprt = next(
            exc.input
            for exc in market.technosphere()
            if exc.input["name"]
            == "electricity voltage transformation from medium to low voltage"
        )

        if df is None:
            data[(imprt.id, market.id, True)] = series
        else:
            total = df.sum(axis=1) + series

            data[(imprt.id, market.id, True)] = series / total
            for exc in market.technosphere():
                if exc.input["name"] in df:
                    data[(exc.input.id, market.id, True)] = (
                        df[exc.input["name"]] / total
                    )

        return data

    def adjust_entso_df_for_medium_voltage(self, df, country):
        """Adjust ENTSO dataframe for use in ecoinvent medium voltage markets.

        1. Get ratio of total hv generation (including T&D losses) to mv production
        2. Convert all ENTSO categories to ecoinvent generators (i.e. disaggregate), using ecoinvent ratios.

        """
        market = self.mv_mixes[country]

        if "Waste" not in df.columns:
            return (
                pd.Series(np.zeros(df.shape[0]), index=df.index),
                pd.Series(np.ones(df.shape[0]), index=df.index),
            )
        if "Solar" in df.columns:
            df["Solar"] *= self.get_ratio_hv_lv_solar_ecoinvent(country)

        hv_generation_with_losses = (
            df.sum(axis=1) - df["Waste"]
        ) / self.mv_loss_coefficients[country]

        if not any(
            exc["amount"]
            for exc in market.technosphere()
            if is_generation_exchange(exc)
            and exc.input["name"] in ENTSO_MAPPING["Waste"]
        ):
            print(
                "Warning with {}, average waste generation of {} but no ecoinvent exchange".format(
                    country, df["Waste"].mean(axis=0)
                )
            )
            return (
                pd.Series(np.zeros(df.shape[0]), index=df.index),
                pd.Series(np.ones(df.shape[0]), index=df.index),
            )
        return df["Waste"].fillna(0), hv_generation_with_losses

    def data_dict_for_medium_voltage_market(
        self, country, years=(2019, 2020, 2021), average=False
    ):
        data = {}

        market = self.mv_mixes[country]
        waste_series, remaining_series = self.adjust_entso_df_for_medium_voltage(
            pd.concat([get_df(country, year) for year in years]), country
        )

        imprt = next(
            exc.input
            for exc in market.technosphere()
            if exc.input["name"]
            == "electricity voltage transformation from high to medium voltage"
        )
        try:
            waste_input = next(
                exc.input
                for exc in market.technosphere()
                if exc.input["name"] in ENTSO_MAPPING["Waste"]
            )
        except StopIteration:
            waste_input = None

        total = waste_series + remaining_series

        data[(imprt.id, market.id, True)] = remaining_series / total
        if waste_input is not None:
            data[(waste_input.id, market.id, True)] = waste_series / total
        return data

    def adjust_entso_df_for_high_voltage(self, df, country):
        """Adjust ENTSO dataframe for use in ecoinvent high voltage markets.

        1. Remove medium voltage waste treatment category.
        2. Adjust solar generation to only include high voltage generation.
        3. Convert all ENTSO categories to ecoinvent generators (i.e. disaggregate), using ecoinvent ratios.

        """
        market = self.hv_mixes[country]

        if "Waste" in df.columns:
            df.drop("Waste", axis=1, inplace=True)
        if "Solar" in df.columns:
            df["Solar"] *= self.get_ratio_hv_lv_solar_ecoinvent(country)

        amounts = {
            exc.input["name"]: exc["amount"]
            for exc in market.technosphere()
            if is_generation_exchange(exc)
        }
        disaggregated = []

        for label, names in ENTSO_MAPPING.items():
            if label not in df.columns:
                continue
            inputs = {name: amounts[name] for name in names if amounts.get(name, 0)}
            if not inputs:
                continue
            subtotal = sum(inputs.values())
            disaggregated.append(
                pd.DataFrame(
                    {
                        name: amount / subtotal * df[label]
                        for name, amount in inputs.items()
                    },
                    index=df.index,
                )
            )
        return pd.concat(disaggregated, axis=1).fillna(0)

    def data_dict_for_high_voltage_market(
        self, country, years=(2019, 2020, 2021), average=False
    ):
        market = self.hv_mixes[country]
        df = self.adjust_entso_df_for_high_voltage(
            pd.concat([get_df(country, year) for year in years]), country
        )

        if country == "CH":
            swissgrid_total_cut = self.swissgrid_total[: df.shape[0]]
            swissgrid_total_cut.index = df.index

        trade = defaultdict(list)
        for src in TRADE_PAIRS[country]:
            for year in years:
                try:
                    trade[src].append(
                        cdc.get_trade(
                            from_country=src,
                            to_country=country,
                            year=year,
                            full_year=True,
                        )
                    )
                except:
                    pass
            if trade[src]:
                trade[src] = pd.concat(trade[src])

                if trade[src].shape[0] != df.shape[0]:
                    trade[src] = trade[src].reindex(df.index).fillna(trade[src].mean())

                trade[src].index = df.index
            else:
                del trade[src]

        if country == "CH":
            total = swissgrid_total_cut + sum(trade.values())
        else:
            if sum(trade.values()).sum():
                total = df.sum(axis=1) + sum(trade.values())
            else:
                total = df.sum(axis=1)

        data = {}

        for exc in market.technosphere():
            if exc.input["name"] in df:
                data[(exc.input.id, market.id, True)] = df[exc.input["name"]] / total
            elif is_trade_exchange(exc) or is_generation_exchange(exc):
                data[(exc.input.id, market.id, False)] = total * 0
        for src, vector in trade.items():
            data[(self.hv_mixes[src].id, market.id, True)] = vector / total

        if country == "CH":
            act = bd.get_activity(("swiss residual electricity mix", "CH-residual"))
            assert act.id == 100000
            data[(act.id, market.id, True)] = (
                swissgrid_total_cut - df.sum(axis=1)
            ) / total

        return data
