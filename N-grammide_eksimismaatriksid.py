import time
from Andmete_sisselaadimine import andmed, nimi
from datetime import datetime
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
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

    for name, model in m:
        algus2 = datetime.now()

        model.fit(X_train, Y_train)

        names = ["eesti", "soome", "vene"]

        disp = plot_confusion_matrix(
            model,
            X_validation,
            Y_validation,
            display_labels=names,
            cmap=pyplot.cm.Blues,
            normalize="true",
        )
        disp.ax_.set_title(f"Normaliseeritud eksimismaatriks {name}")
        pyplot.draw()

        lõpp2 = datetime.now()
        aeg2 = lõpp2 - algus2

        print(
            "{:40s} {:20s}".format(
                f"{name}:",
                f"(Aeg: {str(aeg2)})",
            )
        )

        # pyplot.savefig(
        #     nimi(
        #         "graafikud/{} eksimismaatriks ngramm {} {}"
        #         .format(name, n["tuup"], n["tahis"]),
        #         "png"
        #     ),
        #     bbox_inches="tight",
        #     dpi=100,
        # )

    lõpp = datetime.now()
    aeg = lõpp - algus

    print(f"Aeg: {aeg}\n\n")


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
    {"ngram_size": 4, "df_size": 6, "tuup": "char", "tahis": 0},
    {"ngram_size": 1, "df_size": 2, "tuup": "word", "tahis": 0},
    {"ngram_size": 6, "df_size": 4, "tuup": "char_wb", "tahis": 0},
    {"ngram_size": 5, "df_size": 2, "tuup": "char", "tahis": 1},
    {"ngram_size": 6, "df_size": 4, "tuup": "word", "tahis": 1},
    {"ngram_size": 7, "df_size": 4, "tuup": "char_wb", "tahis": 1},
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