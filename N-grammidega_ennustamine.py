import time

import numpy as np
from Andmete_sisselaadimine import andmed, nimi
from datetime import datetime
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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


def treenimine(m):
    algus = datetime.now()

    print(f"\n\n{n['tuup']} {n['tahis']}")
    print(f"{time.ctime(time.time())}")
    print(f"ngram size: {n['ngram_size']}, df size: {n['df_size']}")
    print("andmepunktid: {0[0]}, tunnused: {0[1]}".format(vektorid.shape))

    results = []
    names = []

    for name, model in m:
        algus2 = datetime.now()

        kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        cv_results = cross_val_score(
            model,
            preprocessing.scale(X_train, with_mean=False),
            np.ravel(Y_train),
            cv=kfold,
            scoring="accuracy",
            n_jobs=-1,
        )
        results.append(cv_results)
        names.append(name)

        print(
            "{:40s} {:3.06f} {:10s} {:20s}".format(
                f"{name}:",
                round(cv_results.mean(), 6),
                "({:1.06f})".format(round(cv_results.std(), 6)),
                f"(Aeg: {str(aeg2)})",
            )
        )

        lõpp2 = datetime.now()
        aeg2 = lõpp2 - algus2

    lõpp = datetime.now()
    aeg = lõpp - algus

    print(f"Aeg: {aeg}\n\n")

    pyplot.boxplot(results, labels=names)
    pyplot.title("Algorithm Comparison / ngramm {} {}".format(n["tuup"], n["tahis"])),
    pyplot.xticks(rotation=90)
    pyplot.draw()

    pyplot.savefig(
        nimi(
            "graafikud/{} eksimismaatriks ngramm {} {}".format(
                name, n["tuup"], n["tahis"]
            ),
            "png",
        ),
        bbox_inches="tight",
        dpi=100,
    )


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

dataset, tekstid = andmed()
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

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        vektorid, keeled, test_size=0.20, random_state=1, shuffle=True
    )

    treenimine(models)

pyplot.show()
pyplot.close()