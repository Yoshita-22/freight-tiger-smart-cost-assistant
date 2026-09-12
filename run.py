from src.freight_tiger_smart_cost_assistant.preprocessing import (
    preprocess_shipments,
)

from src.freight_tiger_smart_cost_assistant.weekly_metrics import (
    calculate_weekly_metrics,
)


# Step 1: preprocess raw shipments
df = preprocess_shipments(
    "data/shipment_records.csv"
)

# Step 2: calculate weekly route metrics
weekly_df = calculate_weekly_metrics(df)


print("\nWeekly data:")
print(weekly_df.head(10).to_string(index=False))


print("\nShape:")
print(weekly_df.shape)