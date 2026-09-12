import pandas as pd


# Number of previous weeks used for the route's historical baseline
HISTORY_WEEKS = 8

# Minimum percentage increase considered unusual.
ANOMALY_THRESHOLD_PCT = 10.0


ROUTE_COLUMNS = [
    "origin",
    "destination",
    "route_type",
]


def calculate_own_history_baseline(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate each route's trailing 8-week average cost.

    Important:
    - Current week is excluded.
    - Only previous available weeks are used.
    - If fewer than 8 previous weeks exist, all available
      previous weeks are used.
    - No padding or extrapolation.
    """

    weekly = weekly.copy()

    # Make sure data is sorted chronologically within each route
    weekly = weekly.sort_values(
        ROUTE_COLUMNS + ["week_of"]
    )

    # Exclude the current week's value first.
    #
    # Example:
    #
    # Week 1   -> NaN
    # Week 2   -> Week 1
    # Week 3   -> Week 2
    # Week 4   -> Week 3
    #
    # Then rolling(8) takes the previous 8 available values.
    weekly["own_history_avg"] = (
        weekly
        .groupby(ROUTE_COLUMNS)["cost_per_tonne_km"]
        .transform(
            lambda x: (
                x
                .shift(1)
                .rolling(
                    window=HISTORY_WEEKS,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    return weekly


def calculate_peer_baseline(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the average cost per tonne-km for other routes
    having the same route_type during the same week.

    The current route is excluded from its own peer average.
    """

    weekly = weekly.copy()

    # First calculate:
    #
    # total cost of all routes with the same
    # route_type + week_of
    #
    # and number of routes in that group.
    peer_stats = (
        weekly
        .groupby(["route_type", "week_of"])["cost_per_tonne_km"]
        .agg(
            peer_total_cost="sum",
            peer_route_count="count",
        )
        .reset_index()
    )

    weekly = weekly.merge(
        peer_stats,
        on=["route_type", "week_of"],
        how="left",
    )

    # Remove the current route from the peer total.
    weekly["peer_cost_excluding_self"] = (
        weekly["peer_total_cost"]
        - weekly["cost_per_tonne_km"]
    )

    weekly["peer_count_excluding_self"] = (
        weekly["peer_route_count"] - 1
    )

    # Calculate peer average.
    #
    # If there are no OTHER routes of the same type
    # in that week, there is no peer baseline.
    weekly["similar_routes_avg"] = (
        weekly["peer_cost_excluding_self"]
        / weekly["peer_count_excluding_self"]
    )

    # Remove helper columns
    weekly = weekly.drop(
        columns=[
            "peer_total_cost",
            "peer_route_count",
            "peer_cost_excluding_self",
            "peer_count_excluding_self",
        ]
    )

    return weekly


def calculate_comparisons(
    weekly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate percentage difference between the current
    weekly cost and the two required baselines.
    """

    weekly = weekly.copy()

    # ---------------------------------------------------------
    # Difference from own historical average
    # ---------------------------------------------------------

    weekly["vs_own_history"] = (
        (
            weekly["cost_per_tonne_km"]
            - weekly["own_history_avg"]
        )
        / weekly["own_history_avg"]
    ) * 100

    # ---------------------------------------------------------
    # Difference from similar routes
    # ---------------------------------------------------------

    weekly["vs_similar_routes"] = (
        (
            weekly["cost_per_tonne_km"]
            - weekly["similar_routes_avg"]
        )
        / weekly["similar_routes_avg"]
    ) * 100

    return weekly


def flag_anomalies(
    weekly: pd.DataFrame,
    threshold_pct: float = ANOMALY_THRESHOLD_PCT,
) -> pd.DataFrame:
    """
    Flag route-weeks whose cost is unusually high compared
    with its own history OR similar routes.

    A route is flagged when either comparison exceeds
    the configured threshold.

    Missing baselines are ignored for that comparison.
    """

    weekly = weekly.copy()

    own_history_anomaly = (
        weekly["vs_own_history"] >= threshold_pct
    )

    similar_route_anomaly = (
        weekly["vs_similar_routes"] >= threshold_pct
    )

    weekly["is_anamalous"] = (
        own_history_anomaly
        | similar_route_anomaly
    )

    # A route with no historical baseline and no peer baseline
    # cannot be classified as anomalous.
    no_baseline = (
        weekly["own_history_avg"].isna()
        & weekly["similar_routes_avg"].isna()
    )

    weekly.loc[no_baseline, "is_anamalous"] = False

    return weekly


def detect_anomalies(
    weekly: pd.DataFrame,
    threshold_pct: float = ANOMALY_THRESHOLD_PCT,
) -> pd.DataFrame:
    """
    Complete anomaly detection pipeline.

    1. Calculate own historical baseline
    2. Calculate similar-route baseline
    3. Calculate percentage differences
    4. Flag anomalies
    """

    weekly = calculate_own_history_baseline(weekly)

    weekly = calculate_peer_baseline(weekly)

    weekly = calculate_comparisons(weekly)

    weekly = flag_anomalies(
        weekly,
        threshold_pct=threshold_pct,
    )

    return weekly