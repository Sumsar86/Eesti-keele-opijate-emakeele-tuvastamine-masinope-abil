import pandas as pd


def andmed():
    url1 = "http://www.tlu.ee/~jaagup/andmed/keel/korpus/dokmeta.txt"
    url2 = "http://www.tlu.ee/~jaagup/andmed/keel/korpus/doksonaliigid.txt"
    names1 = [
        "kood",
        "korpus",
        "tekstikeel",
        "tekstityyp",
        "elukoht",
        "taust",
        "vanus",
        "sugu",
        "emakeel",
        "kodukeel",
        "keeletase",
        "haridus",
        "abivahendid",
    ]
    names2 = [
        "kood",
        "A",
        "C",
        "D",
        "G",
        "H",
        "I",
        "J",
        "K",
        "N",
        "P",
        "S",
        "U",
        "V",
        "X",
        "Y",
        "Z",
        "kokku",
    ]
    dataset1 = pd.read_csv(url1, names=names1)
    dataset2 = pd.read_csv(url2, names=names2)

    dataset = pd.merge(dataset1, dataset2, on="kood")
    dataset = dataset.drop(
        dataset.columns[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]], axis=1
    )

    dataset = dataset.drop(dataset.index[0])
    dataset = dataset.dropna()
    dataset = dataset.apply(pd.to_numeric, errors="ignore")

    dataset = dataset[dataset["kokku"] != 0]

    dataset = dataset.astype(
        {
            "emakeel": str,
            "A": float,
            "C": float,
            "D": float,
            "G": float,
            "H": float,
            "I": float,
            "J": float,
            "K": float,
            "N": float,
            "P": float,
            "S": float,
            "U": float,
            "V": float,
            "X": float,
            "Y": float,
            "Z": float,
            "kokku": float,
        }
    )

    for i, j in dict(dataset["emakeel"].value_counts()).items():
        indeksid = []
        if j < 10 or i == "muud":
            indeksid.append(list(dataset[dataset["emakeel"] == i].index))
        for k in indeksid:
            for t in k:
                dataset.drop(t, inplace=True)

    kokku = dataset["kokku"]
    dataset = dataset.drop(dataset.columns[[-1]], axis=1)

    for i, val in zip(kokku, dataset.iterrows()):
        for j, t in zip(val[1], dataset.columns):
            try:
                dataset.at[val[0], t] = float(j) / i
            except ValueError:
                pass

    dataset = dataset[
        (dataset["emakeel"] == "eesti")
        | (dataset["emakeel"] == "soome")
        | (dataset["emakeel"] == "vene")
    ]
    max_pikkus = min(
        len(dataset[dataset["emakeel"] == "eesti"]),
        len(dataset[dataset["emakeel"] == "soome"]),
        len(dataset[dataset["emakeel"] == "vene"]),
    )

    ds2 = pd.DataFrame()
    ds2 = ds2.append(dataset[dataset["emakeel"] == "eesti"].sample(max_pikkus))
    ds2 = ds2.append(dataset[dataset["emakeel"] == "soome"].sample(max_pikkus))
    ds2 = ds2.append(dataset[dataset["emakeel"] == "vene"].sample(max_pikkus))
    ds2 = ds2.reset_index()
    del ds2["index"]

    return ds2


if __name__ == "__main__":
    ds = andmed()

    print("Andmestiku veerud:")
    [print("   ", i) for i in ds.columns]
    print("\nAndmestiku ridade/tunnuste arv:", len(ds))
    print("\nAndmestik:")
    print(ds.head)