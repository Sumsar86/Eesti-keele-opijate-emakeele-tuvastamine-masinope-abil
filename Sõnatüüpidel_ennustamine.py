from Sõnatüüpide_andmete_sisselaadimine import andmed
import time
import numpy as np
from datetime import datetime
from matplotlib import pyplot
from Ngrammide_andmete_sisselaadimine import nimi
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.svm import SVC, NuSVC


def treenimine(ds, m):
    algus = datetime.now()

    array = ds.values
    x = np.array(array[:, 1:])
    y = np.array(array[:, :1])

    scaler = preprocessing.StandardScaler().fit(x)
    X_scaled = scaler.transform(x)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_scaled, y, test_size=0.20, shuffle=True
    )

    results = []
    names = []

    print(f"\n\nSõnatüüpidel ennustamine")
    print(f"{time.ctime(time.time())}")
    print("andmepunktid: {}, tunnused: 17\n".format(len(X_train)))

    for name, model in m:
        algus2 = datetime.now()
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        try:
            cv_results = cross_val_score(
                model,
                preprocessing.scale(X_train),
                np.ravel(Y_train),
                cv=kfold,
                scoring="accuracy",
                n_jobs=-1,
            )
        except:
            print(
                "================================| Error |================================"
            )

        results.append(cv_results)
        names.append(name)

        lõpp2 = datetime.now()
        aeg2 = lõpp2 - algus2

        print(
            "{:40s} {:3.06f} {:10s} {:20s}".format(
                f"{name}:",
                round(cv_results.mean(), 6),
                "({:1.06f})".format(round(cv_results.std(), 6)),
                f"(Aeg: {str(aeg2)})",
            )
        )

    pyplot.figure(figsize=(8, 6))
    pyplot.boxplot(results, labels=names)
    pyplot.title("Algoritmide täpsused")
    pyplot.xticks(rotation=90)

    pyplot.savefig(
        nimi(
            r"C:\Users\rasmu\OneDrive\Töölaud\Programmid\Python 3\Uurimistöö\Graafikud\Lõplikud\Sõnatüüpide efektiivsus ",
            "png",
        ),
        bbox_inches="tight",
        dpi=100,
    )

    pyplot.draw()

    lõpp = datetime.now()
    aeg = lõpp - algus

    print(f"Aeg: {aeg}\n\n")


# url1 = "http://www.tlu.ee/~jaagup/andmed/keel/korpus/dokmeta.txt"
# url2 = "http://www.tlu.ee/~jaagup/andmed/keel/korpus/doksonaliigid.txt"
# names1 = [
#     "kood",
#     "korpus",
#     "tekstikeel",
#     "tekstityyp",
#     "elukoht",
#     "taust",
#     "vanus",
#     "sugu",
#     "emakeel",
#     "kodukeel",
#     "keeletase",
#     "haridus",
#     "abivahendid",
# ]
# names2 = [
#     "kood",
#     "A",
#     "C",
#     "D",
#     "G",
#     "H",
#     "I",
#     "J",
#     "K",
#     "N",
#     "P",
#     "S",
#     "U",
#     "V",
#     "X",
#     "Y",
#     "Z",
#     "kokku",
# ]
# dataset1 = pd.read_csv(url1, names=names1)
# dataset2 = pd.read_csv(url2, names=names2)

# dataset = pd.merge(dataset1, dataset2, on="kood")
# dataset = dataset.drop(dataset.columns[[0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]], axis=1)

# dataset = dataset.drop(dataset.index[0])
# dataset = dataset.dropna()
# dataset = dataset.apply(pd.to_numeric, errors="ignore")

# dataset = dataset[dataset["kokku"] != 0]

# dataset = dataset.astype(
#     {
#         "emakeel": str,
#         "A": float,
#         "C": float,
#         "D": float,
#         "G": float,
#         "H": float,
#         "I": float,
#         "J": float,
#         "K": float,
#         "N": float,
#         "P": float,
#         "S": float,
#         "U": float,
#         "V": float,
#         "X": float,
#         "Y": float,
#         "Z": float,
#         "kokku": float,
#     }
# )

# for i, j in dict(dataset["emakeel"].value_counts()).items():
#     indeksid = []
#     if j < 10 or i == "muud":
#         indeksid.append(list(dataset[dataset["emakeel"] == i].index))
#     for k in indeksid:
#         for t in k:
#             dataset.drop(t, inplace=True)

# kokku = dataset["kokku"]
# dataset = dataset.drop(dataset.columns[[-1]], axis=1)

# for i, val in zip(kokku, dataset.iterrows()):
#     for j, t in zip(val[1], dataset.columns):
#         try:
#             dataset.at[val[0], t] = float(j) / i
#         except ValueError:
#             pass

# dataset = dataset[
#     (dataset["emakeel"] == "eesti")
#     | (dataset["emakeel"] == "soome")
#     | (dataset["emakeel"] == "vene")
# ]
# max_pikkus = min(
#     len(dataset[dataset["emakeel"] == "eesti"]),
#     len(dataset[dataset["emakeel"] == "soome"]),
#     len(dataset[dataset["emakeel"] == "vene"]),
# )

# ds2 = pd.DataFrame()
# ds2 = ds2.append(dataset[dataset["emakeel"] == "eesti"].sample(max_pikkus))
# ds2 = ds2.append(dataset[dataset["emakeel"] == "soome"].sample(max_pikkus))
# ds2 = ds2.append(dataset[dataset["emakeel"] == "vene"].sample(max_pikkus))
# ds2 = ds2.reset_index()
# del ds2["index"]

ds2 = andmed()


models = [
    (
        (
            "LOGISTIC REGRESSION",
            LogisticRegression(solver="liblinear", multi_class="ovr", max_iter=1000),
        )
    ),
    (("KNEIGHBRS", KNeighborsClassifier())),
    (
        (
            "DECISION TREE CLASSIFIER",
            DecisionTreeClassifier(max_depth=16, min_samples_split=20),
        )
    ),
    (
        (
            "RANDOM FOREST CLASSIFIER",
            RandomForestClassifier(max_depth=16, min_samples_split=20),
        )
    ),
    (("ADABOOST", AdaBoostClassifier(n_estimators=100))),
    (("GRADIENT BOOSTING", GradientBoostingClassifier())),
    (("SGDCLASSIFIER", SGDClassifier(loss="hinge", penalty="l2", max_iter=1500))),
    (("SVM", SVC(gamma="auto"))),
    (("LINEAR DISCRIMINANT ANALASYS", LinearDiscriminantAnalysis())),
    (("QUADRATIC DISCRIMINANT ANALASYS", QuadraticDiscriminantAnalysis())),
    (("GAUSSIAN NB", GaussianNB())),
    (("SVC RBF OVO", SVC(kernel="rbf", decision_function_shape="ovo", gamma="auto"))),
    (("NUSVC RBF", NuSVC(kernel="rbf", gamma="auto"))),
    (("BERNOULLI NB", BernoulliNB())),
    (
        (
            "MLPCLASSIFIER",
            MLPClassifier(
                activation="logistic",
                alpha=1,
                hidden_layer_sizes=(8),
                learning_rate="invscaling",
                solver="adam",
                max_iter=1500,
            ),
        )
    ),
]


treenimine(ds2, models)
pyplot.show()
