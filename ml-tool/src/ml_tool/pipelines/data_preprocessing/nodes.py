"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.5
"""

import pandas as pd


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    labels, uniques = pd.factorize(df["HomePlanet"])
    df["HomePlanetLabeled"] = labels
    labels, uniques = pd.factorize(df["Destination"])
    df["DestinationLabeled"] = labels

    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    labels, uniques = pd.factorize(df["Deck"])
    df["DeckLabeled"] = labels
    labels, uniques = pd.factorize(df["Side"])
    df["SideLabeled"] = labels

    df.fillna(-1, inplace=True)

    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["VIP"] = df["VIP"].astype(int)
    df["Cabin_num"] = df["Cabin_num"].astype(int)

    features = [
        "HomePlanetLabeled",
        "CryoSleep",
        "DestinationLabeled",
        "Age",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "Cabin_num",
        "DeckLabeled",
        "SideLabeled",
    ]

    y = df["Transported"]
    X = df[features]

    return X, y
