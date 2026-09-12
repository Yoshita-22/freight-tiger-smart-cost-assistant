import pandas as pd


# Columns that uniquely identify a route group
ROUTE_COLUMNS = [
    "origin",
    "destination",
    "route_type",
    "week_of",
]


def calculate_weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate shipment records into weekly route-level metrics.

    One output row represents:
        origin + destination + route_type + week_of

    Cost per tonne-km is calculated as:

        total freight cost
        ------------------------------
        total(quantity × distance)
    """

    df = df.copy()

    # Calculate the denominator contribution for each shipment.
    #
    # For each shipment:
    # quantity_tonnes × distance_km
    #
    # We calculate this before grouping because distance can vary
    # slightly between shipments on the same route.
    df["tonne_km"] = (
        df["quantity_tonnes"] * df["distance_km"]
    )

    # Aggregate shipments by route + route type + week
    weekly = (
        df.groupby(ROUTE_COLUMNS, as_index=False)
        .agg(
            total_freight_cost_inr=(
                "freight_cost_inr",
                "sum",
            ),
            total_tonne_km=(
                "tonne_km",
                "sum",
            ),
            shipment_count=(
                "shipment_id",
                "count",
            ),
        )
    )

    # Calculate weekly cost per tonne-km
    weekly["cost_per_tonne_km"] = (
        weekly["total_freight_cost_inr"]
        / weekly["total_tonne_km"]
    )

    # Sort chronologically
    weekly = weekly.sort_values(
        ["origin", "destination", "route_type", "week_of"]
    ).reset_index(drop=True)

    return weekly