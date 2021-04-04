import os
import pandas as pd
import glob


def nimi(koht, lõpp):
    i = 0
    while os.path.exists(rf"{koht} {i}.{lõpp}"):
        i += 1
    return rf"{koht} {i}.{lõpp}"

def andmed():

    # Andmete (teksti kood, autori emakeel ja tekstikeel) internetist alla laadimine ja Pandase
    # DataFrame objekti panemine ning üleliigsete veergude eemaldamine.

    url = "http://www.tlu.ee/~jaagup/andmed/keel/korpus/dokmeta.txt"

    names = [
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

    dataset = pd.read_csv(url, names=names)

    dataset = dataset.drop(dataset.columns[[1, 3, 4, 5, 6, 7, 9, 10, 11, 12]], axis=1)
    dataset = dataset.drop(dataset.index[0])

    dataset = dataset.astype({"kood": str, "emakeel": str, "tekstikeel": str})

    dataset = dataset[
        (dataset["emakeel"] == "eesti")
        | (dataset["emakeel"] == "soome")
        | (dataset["emakeel"] == "vene")
    ]
    dataset = dataset[dataset["tekstikeel"] == "eesti"]

    max_pikkus = min(
        len(dataset[dataset["emakeel"] == "eesti"]),
        len(dataset[dataset["emakeel"] == "soome"]),
        len(dataset[dataset["emakeel"] == "vene"]),
    )

    ds = pd.DataFrame()
    ds = ds.append(dataset[dataset["emakeel"] == "eesti"].sample(max_pikkus))
    ds = ds.append(dataset[dataset["emakeel"] == "soome"].sample(max_pikkus))
    ds = ds.append(dataset[dataset["emakeel"] == "vene"].sample(max_pikkus))
    ds = ds.reset_index()
    del ds["index"]

    # Tekstid on juba alla laeutd ning asuvad kasutas "koik_dokumendid", kust neid ükshaaval loetakse,
    # seotakse koodi järgi eelnevate andmetega ja pannakse edasiseks kasutamiseks uude DataFrame objekti.

    tekstid = []
    koodid = []
    oiged_margid = (
        "q",
        "w",
        "e",
        "r",
        "t",
        "y",
        "u",
        "i",
        "o",
        "p",
        "ü",
        "õ",
        "a",
        "s",
        "d",
        "f",
        "g",
        "h",
        "j",
        "k",
        "l",
        "ö",
        "ä",
        "z",
        "x",
        "c",
        "v",
        "b",
        "n",
        "m",
        "ž",
        "š",
        " ",
    )
    # failid = glob.glob(r"koik_dokumendid\*.txt")
    failid = glob.glob(
        r"C:\Users\rasmu\OneDrive\Töölaud\Programmid\Python 3\Uurimistöö\koik_dokumendid\*.txt"
    )

    # Tekstide töötlemine. Haruldased või tavapäratud ASCII märgid eemaldatakse tekstidest.

    for i in range(len(failid)):
        if failid[i][-25:-4] in set(ds["kood"]):
            f = open(failid[i], "r")
            x = f.read().replace("\n", " ")
            for y in x:
                if y.lower() in oiged_margid:
                    x.replace(y, "")
            if x != "":
                tekstid.append(x)
                koodid.append([x, failid[i][-25:-4]])
            f.close()

    toodeldud_failid = pd.DataFrame(koodid, columns=["tekst", "kood"])
    uus_dataset = pd.merge(dataset, toodeldud_failid, on="kood")
    uus_dataset = uus_dataset.drop(uus_dataset.columns[[1]], axis=1)

    return uus_dataset, tekstid


if __name__ == "__main__":
    ds = andmed()[0]

    print("Andmestiku veerud:")
    [print("   ", i) for i in ds.columns]
    print("\nAndmestiku ridade/tunnuste arv:", len(ds))
    print("\nAndmestik:")
    print(ds.head)