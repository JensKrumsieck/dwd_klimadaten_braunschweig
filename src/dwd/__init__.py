from wetterdienst.provider.dwd.observation import DwdObservationRequest
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots

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
) -> None:
    data = _combine_stations(df, columns, time_col)
    with plt.rc_context({"text.usetex": False}):
        fig, ax = plt.subplots(figsize=(10, 4))
        for column in columns:
            series = data.drop_nulls(column)
            ax.plot(
                series[time_col],
                series[column],
                marker="o",
                markersize=3,
                label=(labels or {}).get(column, column),
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
        "Heiße Tage & Sommertage – jährlich",
        "count_days_yearly.png",
        annotate=(1953, note),
        labels=labels,
    )
    plot_lines(
        df_decade,
        columns,
        "decade",
        "Anzahl Tage",
        "Heiße Tage & Sommertage – je Dekade",
        "count_days_decade.png",
        annotate=(1950, note),
        labels=labels,
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
        "Temperaturen – jährlich",
        "temperatures_yearly.png",
        annotate=(1953, note),
        labels=labels,
    )
    plot_lines(
        df_decade,
        columns,
        "decade",
        "Temperatur / °C",
        "Temperaturen – je Dekade",
        "temperatures_decade.png",
        annotate=(1950, note),
        labels=labels,
    )
    last_50_years = df_yearly.filter(pl.col("year") >= df_yearly["year"].max() - 49)
    plot_lines(
        last_50_years,
        columns,
        "year",
        "Temperatur / °C",
        "Temperaturen – letzte 50 Jahre",
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
        "Temperaturen – Dekaden der letzten 50 Jahre",
        "temperatures_decade_last50y.png",
        labels=labels,
    )


def main():
    df_yearly = get_yearly_observations()
    df_yearly.to_pandas().to_csv("dwd_yearly_observations.csv", index=False)
    plot_warming_stripes(df_yearly, save_path="warming_stripes.png")
    plot_warming_stripes_uniform(df_yearly, save_path="warming_stripes_uniform.png")

    df_decade = get_decade_observations(df_yearly)
    df_decade.to_pandas().to_csv("dwd_decade_observations.csv", index=False)

    plot_count_days(df_yearly, df_decade)
    plot_temperatures(
        df_yearly,
        df_decade,
    )


if __name__ == "__main__":
    main()
