'''
This code shows how to write an ANN to predict the median_house_value,
other remaining features are used for training
'''

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

######################
## Data preparation ##
######################

import polars as pl
from polars import col as c

pl_housing = (
    pl.read_csv(source="https://raw.githubusercontent.com/dmarks84/Ind_Project_California-Housing-Data--Kaggle/refs/heads/main/housing.csv")
    .to_dummies(columns="ocean_proximity", drop_first=True)
    .with_columns(total_bedrooms = c("total_bedrooms").fill_null(pl.col("total_bedrooms").median()))
    .cast(pl.Float32)
)
print(pl_housing.head())
# ┌─────────────┬───────────┬────────────────────┬─────────────┬───┬─────────────────────┬────────────────────────┬────────────────────────┬────────────────────────────┐
# │ longitude   ┆ latitude  ┆ housing_median_age ┆ total_rooms ┆ … ┆ ocean_proximity_<1H ┆ ocean_proximity_INLAND ┆ ocean_proximity_ISLAND ┆ ocean_proximity_NEAR OCEAN │
# │ ---         ┆ ---       ┆ ---                ┆ ---         ┆   ┆ OCEAN               ┆ ---                    ┆ ---                    ┆ ---                        │
# │ f32         ┆ f32       ┆ f32                ┆ f32         ┆   ┆ ---                 ┆ f32                    ┆ f32                    ┆ f32                        │
# │             ┆           ┆                    ┆             ┆   ┆ f32                 ┆                        ┆                        ┆                            │
# ╞═════════════╪═══════════╪════════════════════╪═════════════╪═══╪═════════════════════╪════════════════════════╪════════════════════════╪════════════════════════════╡
# │ -122.230003 ┆ 37.880001 ┆ 41.0               ┆ 880.0       ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.220001 ┆ 37.860001 ┆ 21.0               ┆ 7099.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.239998 ┆ 37.849998 ┆ 52.0               ┆ 1467.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.25     ┆ 37.849998 ┆ 52.0               ┆ 1274.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# │ -122.25     ┆ 37.849998 ┆ 52.0               ┆ 1627.0      ┆ … ┆ 0.0                 ┆ 0.0                    ┆ 0.0                    ┆ 0.0                        │
# └─────────────┴───────────┴────────────────────┴─────────────┴───┴─────────────────────┴────────────────────────┴────────────────────────┴────────────────────────────┘

print(pl_housing.shape)
# (20640, 13)

'''----- X ------'''
X = torch.tensor(
    data=pl_housing.select(c("*").exclude("median_house_value")).to_numpy(),
    device=device
)
print(X)
# tensor([[-122.2300,   37.8800,   41.0000,  ...,    0.0000,    0.0000,
#             0.0000],
#         [-122.2200,   37.8600,   21.0000,  ...,    0.0000,    0.0000,
#             0.0000],
#         [-122.2400,   37.8500,   52.0000,  ...,    0.0000,    0.0000,
#             0.0000],
#         ...,
#         [-121.2200,   39.4300,   17.0000,  ...,    1.0000,    0.0000,
#             0.0000],
#         [-121.3200,   39.4300,   18.0000,  ...,    1.0000,    0.0000,
#             0.0000],
#         [-121.2400,   39.3700,   16.0000,  ...,    1.0000,    0.0000,
#             0.0000]], device='cuda:0')

'''----- y ------'''
y = torch.tensor(
    data=pl_housing.get_column("median_house_value").to_numpy(),
    device=device
)
print(y)
# tensor([452600., 358500., 352100.,  ...,  92300.,  84700.,  89400.],
#        device='cuda:0')