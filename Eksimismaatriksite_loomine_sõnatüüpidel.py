import time
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot
from Ngrammide_andmete_sisselaadimine import nimi
from Sõnatüüpide_andmete_sisselaadimine import andmed
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics._plot.confusion_matrix import plot_confusion_matrix
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


def funktsioon(ds, m):
    algus = datetime.now()

    array = ds.values
    x = np.array(array[:, 1:])
    y = np.array(array[:, :1])

    scaler = preprocessing.StandardScaler().fit(x)
    X_scaled = scaler.transform(x)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_scaled, y, test_size=0.20, shuffle=True
    )

    print("\n\nsõnatüübid")
    print(f"{time.ctime(time.time())}")
    print("andmepunktid: {}, tunnused: 17\n".format(len(X_train)))

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
        pyplot.savefig(
            nimi("graafikud/Normaliseeritud eksimismaatriks {name} ", "png"),
            bbox_inches="tight",
            dpi=100,
        )

        lõpp2 = datetime.now()
        aeg2 = lõpp2 - algus2

        print(
            "{:40s} {:150s} {:20s}".format(
                f"{name}:",
                f"{list(disp.confusion_matrix)}",
                f"(Aeg: {str(aeg2)})",
            )
        )

    lõpp = datetime.now()
    aeg = lõpp - algus

    print(f"Aeg: {aeg}\n\n")


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

funktsioon(ds2, models)
pyplot.show()
