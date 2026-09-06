import io
import urllib.request

from wetterdienst.provider.dwd.observation import DwdObservationRequest
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

GERMANY_REGIONAL_AVERAGES_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/regional_averages_DE/"
    "annual/air_temperature_mean/regional_averages_tm_year.txt"
)

plt.style.use(["science", "no-latex", "nature"])


def get_dwd_observations() -> pl.DataFrame:
    request = DwdObservationRequest(
        # fmt: off
       parameters=[
            # temperatures (climate_summary)
            ("annual", "climate_summary", "temperature_air_mean_2m"),       # mean of daily means
            ("annual", "climate_summary", "temperature_air_max_2m_mean"),   # mean of daily maxima
            ("annual", "climate_summary", "temperature_air_min_2m_mean"),   # mean of daily minima
            ("annual", "climate_summary", "temperature_air_max_2m"),        # hottest day of the year
            ("annual", "climate_summary", "temperature_air_min_2m"),        # coldest day of the year
            # other climate_summary values
            ("annual", "climate_summary", "precipitation_height"),
            ("annual", "climate_summary", "sunshine_duration"),
            ("annual", "climate_summary", "cloud_cover_total"),
            ("annual", "climate_summary", "wind_force_beaufort"),
            # day-count indices (climate_indices)
            ("annual", "climate_indices", "count_days_frost"),
            ("annual", "climate_indices", "count_days_ice"),
            ("annual", "climate_indices", "count_days_summer"),
            ("annual", "climate_indices", "count_days_hot"),
            ("annual", "climate_indices", "count_days_tropical_night"),
        ],
        # fmt: on
        start_date="1800-01-01",
        end_date="2026-09-04",
    ).filter_by_station_id(station_id=[661, 662])
    df = request.values.all().df
    return df


def get_yearly_observations() -> pl.DataFrame:
    df = get_dwd_observations()
    return (
        df.with_columns(pl.col("date").dt.year().alias("year"))
        .pivot(on="parameter", index=["station_id", "year"], values="value")
        .sort("station_id", "year")
    )


def get_decade_observations(df_yearly: pl.DataFrame | None = None) -> pl.DataFrame:
    df = df_yearly if df_yearly is not None else get_yearly_observations()
    return (
        df.with_columns((pl.col("year") // 10 * 10).alias("decade"))
        .group_by("station_id", "decade")
        .agg(pl.exclude("station_id", "year", "decade").mean())
        .sort("station_id", "decade")
    )


def get_germany_yearly_temperature() -> pl.DataFrame:
    request = urllib.request.Request(
        GERMANY_REGIONAL_AVERAGES_URL, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode("latin-1")
    df = pl.read_csv(
        io.StringIO(text), separator=";", skip_rows=1, truncate_ragged_lines=True
    )
    return df.select(
        pl.col("Jahr").alias("year"),
        pl.col("Deutschland")
        .str.strip_chars()
        .cast(pl.Float64)
        .alias("temperature_germany"),
        pl.col("Niedersachsen")
        .str.strip_chars()
        .cast(pl.Float64)
        .alias("temperature_niedersachsen"),
    )


def get_germany_decade_temperature(
    df_germany_yearly: pl.DataFrame | None = None,
) -> pl.DataFrame:
    df = (
        df_germany_yearly
        if df_germany_yearly is not None
        else get_germany_yearly_temperature()
    )
    return (
        df.with_columns((pl.col("year") // 10 * 10).alias("decade"))
        .group_by("decade")
        .agg(pl.exclude("year", "decade").mean())
        .sort("decade")
    )


HANNOVER_STATION_ID = "02014"


def get_hannover_yearly_temperature() -> pl.DataFrame:
    request = DwdObservationRequest(
        parameters=[("annual", "climate_summary", "temperature_air_mean_2m")],
        start_date="1800-01-01",
        end_date="2026-09-04",
    ).filter_by_station_id(station_id=[HANNOVER_STATION_ID])
    df = request.values.all().df
    return (
        df.with_columns(pl.col("date").dt.year().alias("year"))
        .select("year", pl.col("value").alias("temperature_hannover"))
        .sort("year")
    )


def get_hannover_decade_temperature(
    df_hannover_yearly: pl.DataFrame | None = None,
) -> pl.DataFrame:
    df = (
        df_hannover_yearly
        if df_hannover_yearly is not None
        else get_hannover_yearly_temperature()
    )
    return (
        df.with_columns((pl.col("year") // 10 * 10).alias("decade"))
        .group_by("decade")
        .agg(pl.exclude("year", "decade").mean())
        .sort("decade")
    )


def plot_warming_stripes(
    df: pl.DataFrame, save_path: str = "warming_stripes.png"
) -> None:
    df = (
        df.group_by("year")
        .agg(pl.col("temperature_air_mean_2m").mean())
        .drop_nulls("temperature_air_mean_2m")
        .sort("year")
    )
    years = df["year"].to_numpy()
    temps = df["temperature_air_mean_2m"].to_numpy()
    baseline = df.filter(pl.col("year").is_between(1971, 2000))[
        "temperature_air_mean_2m"
    ]
    baseline_mean = baseline.mean()
    anomaly = temps - baseline_mean
    limit = max(abs(anomaly.min()), abs(anomaly.max()))

    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(len(years) * 0.08, 4))
        ax.bar(
            years,
            anomaly,
            width=1.0,
            color=plt.cm.RdBu_r(anomaly / (4 * temps.std()) + 0.5),
        )

        ax.title.set_text(
            "Jahresmitteltemperatur (Anomalie) im Vergleich zum Mittelwert 1971-2000\nDatenquelle: Deutscher Wetterdienst (DWD) - Stationsmessungen für Braunschweig (Station 661 und 662)"
        )
        ax.set_ylabel("Anomalie / °C")
        decade_start = (years.min() // 10 + 1) * 10
        ax.set_xticks(range(decade_start, years.max() + 1, 10))
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_ylim(-limit - 0.1, limit + 0.1)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def plot_warming_stripes_uniform(
    df: pl.DataFrame, save_path: str = "warming_stripes_uniform.png"
) -> None:
    df = (
        df.group_by("year")
        .agg(pl.col("temperature_air_mean_2m").mean())
        .drop_nulls("temperature_air_mean_2m")
        .sort("year")
    )
    years = df["year"].to_numpy()
    temps = df["temperature_air_mean_2m"].to_numpy()
    baseline = df.filter(pl.col("year").is_between(1971, 2000))[
        "temperature_air_mean_2m"
    ]
    anomaly = temps - baseline.mean()

    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(len(years) * 0.08, 4))
        ax.bar(
            years,
            [1] * len(years),
            width=1.0,
            color=plt.cm.RdBu_r(anomaly / (4 * temps.std()) + 0.5),
        )
        ax.set_axis_off()
        ax.set_xlim(years.min() - 0.5, years.max() + 0.5)
        ax.set_ylim(0, 1)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)


def _combine_stations(
    df: pl.DataFrame, columns: list[str], time_col: str
) -> pl.DataFrame:
    return df.group_by(time_col).agg([pl.col(c).mean() for c in columns]).sort(time_col)


def plot_lines(
    df: pl.DataFrame,
    columns: list[str],
    time_col: str,
    ylabel: str,
    title: str,
    save_path: str,
    annotate: tuple[float, str] | None = None,
    labels: dict[str, str] | None = None,
    mask_x: float | None = None,
    mask_alpha: float = 0.35,
    mask_columns: list[str] | None = None,
) -> None:
    data = _combine_stations(df, columns, time_col)
    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(10, 4))
        for column in columns:
            series = data.drop_nulls(column)
            apply_mask = mask_x is not None and (
                mask_columns is None or column in mask_columns
            )
            time_values = series[time_col].to_list()
            markevery = [i for i, x in enumerate(time_values) if x != mask_x]
            (line,) = ax.plot(
                series[time_col],
                series[column],
                marker="o",
                markersize=3,
                markevery=markevery if apply_mask else None,
                label=(labels or {}).get(column, column),
            )
            if apply_mask and mask_x in time_values:
                masked = series.filter(pl.col(time_col) == mask_x)
                ax.plot(
                    masked[time_col],
                    masked[column],
                    marker="o",
                    markersize=3,
                    color=line.get_color(),
                    alpha=mask_alpha,
                )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        if annotate is not None:
            x, text = annotate
            ax.axvline(x, color="grey", linestyle="--", linewidth=1)
            ax.annotate(
                text,
                xy=(x, 1),
                xycoords=("data", "axes fraction"),
                xytext=(5, -5),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=8,
                color="grey",
            )
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def plot_count_days2(df_yearly, df_decade) -> None:
    columns = ["count_days_hot"]
    labels = {
        "count_days_hot": "Heiße Tage (Tmax ≥ 30 °C)",
    }
    note = "Stationswechsel: Innenstadt → Stadtrand"
    plot_lines(
        df_yearly,
        columns,
        "year",
        "Anzahl Tage",
        "Heiße Tage - jährlich",
        "count_days_yearly2.png",
        annotate=(1953, note),
        labels=labels,
    )
    plot_lines(
        df_decade,
        columns,
        "decade",
        "Anzahl Tage",
        "Heiße Tage - je Dekade",
        "count_days_decade2.png",
        annotate=(1950, note),
        labels=labels,
        mask_x=1950,
    )


def plot_count_days(df_yearly, df_decade) -> None:
    columns = ["count_days_hot", "count_days_summer"]
    labels = {
        "count_days_hot": "Heiße Tage (Tmax ≥ 30 °C)",
        "count_days_summer": "Sommertage (Tmax ≥ 25 °C)",
    }
    note = "Stationswechsel: Innenstadt → Stadtrand"
    plot_lines(
        df_yearly,
        columns,
        "year",
        "Anzahl Tage",
        "Heiße Tage & Sommertage - jährlich",
        "count_days_yearly.png",
        annotate=(1953, note),
        labels=labels,
    )
    plot_lines(
        df_decade,
        columns,
        "decade",
        "Anzahl Tage",
        "Heiße Tage & Sommertage - je Dekade",
        "count_days_decade.png",
        annotate=(1950, note),
        labels=labels,
        mask_x=1950,
    )


def plot_temperatures(df_yearly, df_decade) -> None:
    columns = ["temperature_air_mean_2m"]
    labels = {"temperature_air_mean_2m": "Jahresmitteltemperatur"}
    note = "Stationswechsel: Innenstadt → Stadtrand"
    plot_lines(
        df_yearly,
        columns,
        "year",
        "Temperatur / °C",
        "Temperaturen - jährlich",
        "temperatures_yearly.png",
        annotate=(1953, note),
        labels=labels,
    )
    plot_lines(
        df_decade,
        columns,
        "decade",
        "Temperatur / °C",
        "Temperaturen - je Dekade",
        "temperatures_decade.png",
        annotate=(1950, note),
        labels=labels,
        mask_x=1950,
    )
    last_50_years = df_yearly.filter(pl.col("year") >= df_yearly["year"].max() - 49)
    plot_lines(
        last_50_years,
        columns,
        "year",
        "Temperatur / °C",
        "Temperaturen - letzte 50 Jahre",
        "temperatures_last50y.png",
        labels=labels,
    )
    last_5_decades = df_decade.filter(
        pl.col("decade") >= df_decade["decade"].max() - 40
    )
    plot_lines(
        last_5_decades,
        columns,
        "decade",
        "Temperatur / °C",
        "Temperaturen - Dekaden der letzten 50 Jahre",
        "temperatures_decade_last50y.png",
        labels=labels,
    )


def plot_temperature_comparison(
    df_decade: pl.DataFrame,
    df_germany_decade: pl.DataFrame,
    df_hannover_decade: pl.DataFrame,
    save_path: str = "temperature_comparison_decade.png",
) -> None:
    braunschweig = _combine_stations(df_decade, ["temperature_air_mean_2m"], "decade")
    combined = braunschweig.join(df_germany_decade, on="decade", how="inner").join(
        df_hannover_decade, on="decade", how="inner"
    )
    columns = [
        "temperature_air_mean_2m",
        "temperature_hannover",
        "temperature_niedersachsen",
        "temperature_germany",
    ]
    labels = {
        "temperature_air_mean_2m": "Braunschweig",
        "temperature_hannover": "Hannover",
        "temperature_niedersachsen": "Niedersachsen",
        "temperature_germany": "Deutschland",
    }
    plot_lines(
        combined,
        columns,
        "decade",
        "Temperatur / °C",
        "Jahresmitteltemperatur je Dekade - Braunschweig vs. Hannover vs. Niedersachsen vs. Deutschland",
        save_path,
        annotate=(1950, "Stationswechsel: Innenstadt → Stadtrand"),
        labels=labels,
        mask_x=1950,
        mask_columns=["temperature_air_mean_2m"],
    )


def plot_temperature_regression(
    df_decade: pl.DataFrame,
    save_path: str = "temperature_regression_decade.png",
    fit_from: int = 1960,
    xlim: tuple[int, int] = (1960, 2050),
) -> None:
    data = _combine_stations(
        df_decade, ["temperature_air_mean_2m"], "decade"
    ).drop_nulls("temperature_air_mean_2m")
    fit_data = data.filter(pl.col("decade") >= fit_from)
    decades = fit_data["decade"].to_numpy()
    temps = fit_data["temperature_air_mean_2m"].to_numpy()
    slope, intercept = np.polyfit(decades, temps, 1)

    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            fit_data["decade"],
            fit_data["temperature_air_mean_2m"],
            marker="o",
            markersize=4,
            linestyle="none",
            color="tab:blue",
            label="Braunschweig - Jahresmitteltemperatur je Dekade",
        )
        regression_x = np.array(xlim)
        ax.plot(
            regression_x,
            slope * regression_x + intercept,
            linestyle="--",
            color="tab:red",
            label="lineare Regression",
        )
        ax.set_xlim(*xlim)
        ax.set_ylabel("Temperatur / °C")
        ax.set_title(
            "Braunschweig - Trend der Jahresmitteltemperatur je Dekade (ab 1960)"
        )
        ax.legend()
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)


def main():
    df_yearly = get_yearly_observations()
    df_yearly.to_pandas().to_csv("dwd_yearly_observations.csv", index=False)
    plot_warming_stripes(df_yearly, save_path="warming_stripes.png")
    plot_warming_stripes_uniform(df_yearly, save_path="warming_stripes_uniform.png")

    df_decade = get_decade_observations(df_yearly)
    df_decade.to_pandas().to_csv("dwd_decade_observations.csv", index=False)

    plot_count_days(df_yearly, df_decade)
    plot_count_days2(df_yearly, df_decade)
    plot_temperatures(
        df_yearly,
        df_decade,
    )

    df_germany_yearly = get_germany_yearly_temperature()
    df_germany_yearly.to_pandas().to_csv("germany_yearly_temperature.csv", index=False)
    df_germany_decade = get_germany_decade_temperature(df_germany_yearly)

    df_hannover_yearly = get_hannover_yearly_temperature()
    df_hannover_yearly.to_pandas().to_csv(
        "hannover_yearly_temperature.csv", index=False
    )
    df_hannover_decade = get_hannover_decade_temperature(df_hannover_yearly)

    plot_temperature_comparison(df_decade, df_germany_decade, df_hannover_decade)
    plot_temperature_regression(df_decade)


if __name__ == "__main__":
    main()
