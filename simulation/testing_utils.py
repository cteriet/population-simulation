import polars as pl

test_df = pl.DataFrame(
    {
        "intercept": [1, 1, 1, 1, 1],
        "customer_type_B": [1, 0, 1, 0, 1],
        "region_east": [0, 1, 0, 1, 0],
        "Marketing_Coffee": [1, 0, 0, 0, 0],
        "Group_HotBeverage": [1, 1, 0, 0, 0],
        "Marketing_Tea": [0, 1, 0, 0, 0],
        "Group_ColdBeverage": [0, 0, 1, 1, 1],
        "Marketing_Water": [0, 0, 1, 0, 0],
        "Marketing_IcedTea": [0, 0, 0, 1, 0],
        "Marketing_Cola": [0, 0, 0, 0, 1],
    }
)
