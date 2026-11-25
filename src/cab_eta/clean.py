"""Clean raw MP054 training records without changing their source file."""

NUMERIC_FEATURES = [
    "drivers_nearby", "distance_to_driver_km", "hour",
    "rain_mm", "surge", "road_closures",
]
CATEGORICAL_FEATURES = ["car_type", "zone"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_training(raw, findings, target):
    cleaned = raw.drop_duplicates().copy()
    cleaned["date"] = cleaned["date"].astype("datetime64[ns]")

    unit = findings["unit_mixup"]
    if unit["present"]:
        column = unit["column"]
        cleaned[column] = cleaned[column].astype(float)
        affected = cleaned["zone"].eq(unit["group_value"])
        cleaned.loc[affected, column] /= unit["factor"]

    sentinel = findings["sentinel_value"]
    if sentinel["present"]:
        column = sentinel["column"]
        cleaned[column] = cleaned[column].mask(cleaned[column].eq(sentinel["value"]))

    typos = findings["target_typos"]
    if typos["present"]:
        # The audit identifies the upper tail before duplicate removal.
        affected = raw[target].nlargest(typos["count"]).index
        cleaned[target] = cleaned[target].astype(float)
        rows = cleaned.index.intersection(affected)
        cleaned.loc[rows, target] /= 10

    cleaned["car_type"] = (
        cleaned["car_type"].str.strip().str.lower()
        .replace({"aut.": "auto", "min.": "mini", "sed.": "sedan"})
    )
    return cleaned.sort_values("date", kind="stable").reset_index(drop=True)
