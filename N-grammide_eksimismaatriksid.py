import time
import pandas as pd
import seaborn as sn
from datetime import datetime
from matplotlib import pyplot
from Andmete_sisselaadimine import andmed, nimi
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import NuSVC
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)


def treenimine(mud, vek, kee, n):
    algus = datetime.now()

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        vek, kee, test_size=0.20, random_state=1, shuffle=True
    )

    results = []

    print(f"\n\nN-grammide eksimismaatriksite loomine {n['tuup']} {n['tahis']}")
    print(f"{time.ctime(time.time())}")
    print(f"ngram size: {n['ngram_size']}, df size: {n['df_size']}")
    print("andmepunktid: {0[0]}, tunnused: {0[1]}".format(vek.shape))

    for name, model in mud:
        algus2 = datetime.now()

        model.fit(X_train, Y_train)
        pred = model.predict(X_validation)

        results.append(confusion_matrix(Y_validation, pred, normalize="true"))

        lõpp2 = datetime.now()
        aeg2 = lõpp2 - algus2

        print(
            "{:40s} {:20s}".format(
                f"{name}:",
                f"(Aeg: {str(aeg2)})",
            )
        )

    lõpp = datetime.now()
    aeg = lõpp - algus

    print(f"Aeg: {aeg}\n\n")

    return results


dataset, tekstid = andmed()
tulemused = []

models = [
    (
        "LOGISTIC REGRESSION",
        LogisticRegression(
            C=0.1,
            penalty="l2",
            solver="liblinear",
            tol=1e-10,
            multi_class="auto",
            max_iter=1500,
        ),
    ),
    ("KNEIGHBRS", KNeighborsClassifier()),
    (
        "DECISION TREE CLASSIFIER/CART",
        DecisionTreeClassifier(max_depth=16, min_samples_split=20),
    ),
    (
        "RANDOM FOREST CLASSIFIER",
        RandomForestClassifier(max_depth=16, min_samples_split=20),
    ),
    ("ADABOOST", AdaBoostClassifier(learning_rate=1.0, n_estimators=100)),
    ("GRADIENT BOOSTING", GradientBoostingClassifier()),
    ("SGDCLASSIFIER", SGDClassifier(loss="log", penalty="l2", max_iter=1500)),
    ("NUSVC RBF", NuSVC(kernel="rbf", gamma="auto")),
    ("MULTINOMAL NB", MultinomialNB()),
    ("COMPLEMENT NB", ComplementNB()),
    ("BERNOULLI NB", BernoulliNB()),
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
    ),
]
ngrammid = [
    {"ngram_size": 4, "df_size": 6, "tahis": 0, "tuup": "char"},
    {"ngram_size": 1, "df_size": 2, "tahis": 0, "tuup": "word"},
    {"ngram_size": 6, "df_size": 4, "tahis": 0, "tuup": "char_wb"},
    {"ngram_size": 5, "df_size": 2, "tahis": 1, "tuup": "char"},
    {"ngram_size": 6, "df_size": 4, "tahis": 1, "tuup": "word"},
    {"ngram_size": 7, "df_size": 4, "tahis": 1, "tuup": "char_wb"},
]

for n in ngrammid:
    if n["tahis"] == 0:
        xvektorid = CountVectorizer(
            ngram_range=(n["ngram_size"], n["ngram_size"]),
            min_df=n["df_size"],
            analyzer=n["tuup"],
        )
    else:
        xvektorid = CountVectorizer(
            ngram_range=(0, n["ngram_size"]), min_df=n["df_size"], analyzer=n["tuup"]
        )

    xvektorid.fit(tekstid)
    vektorid = xvektorid.transform(tekstid)
    vektorid.toarray()

    keeled = dataset["emakeel"].to_numpy()

    tulemused.append(treenimine(models, vektorid, keeled, n))


nimed = [
    "LOGISTIC REGRESSION",
    "KNEIGHBRS",
    "DECISION TREE CLASSIFIER",
    "RANDOM FOREST CLASSIFIER",
    "ADABOOST",
    "GRADIENT BOOSTING",
    "SGDCLASSIFIER",
    "NUSVC RBF",
    "MULTINOMAL NB",
    "COMPLEMENT NB",
    "BERNOULLI NB",
    "MLPCLASSIFIER",
]
variandid = ["char_wb 0", "char_wb 1", "char 0", "char 1", "word 0", "word 1"]
keeled = ["eesti", "soome", "vene"]
uus_tulemused = [[] for _ in range(len(models))]

for i in range(len(tulemused)):
    for j in range(len(tulemused[i])):
        uus_tulemused[j].append(tulemused[i][j])

laius = 3
korgus = 2

for j in range(len(uus_tulemused)):
    fig, axs = pyplot.subplots(2, laius, figsize=(15, 10))
    fig.suptitle(f"Normaliseeritud eksimismaatriks {nimed[j]}")
    for i in range(len(uus_tulemused[0])):
        axs[i // laius, i % laius].set_title(variandid[i])
        df_cm = pd.DataFrame(uus_tulemused[j][i], index=keeled, columns=keeled)
        sn.heatmap(
            df_cm, annot=True, cmap=pyplot.cm.Blues, ax=axs[i // laius, i % laius]
        )
    pyplot.draw()
    pyplot.savefig(
        nimi(
            rf"C:\Users\rasmu\OneDrive\Töölaud\Programmid\Python 3\Uurimistöö\Graafikud\Normaliseeritud eksimismaatriks {nimed[j]} ngrammid",
            "png",
        ),
        bbox_inches="tight",
        dpi=100,
    )

pyplot.show()
pyplot.close()