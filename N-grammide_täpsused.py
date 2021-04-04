from Andmete_sisselaadimine import andmed
from datetime import datetime
from matplotlib.colors import Normalize
import numpy as np
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def andmete_tootlus(jar=None, x=None, y=None):
    if jar != None:
        if len(jar) < 2:
            return jar[0]
        if len(jar) > 2:
            x = andmete_tootlus(jar=jar[1:])
            y = jar[0]
        else:
            x = jar[1]
            y = jar[0]

    tul = [[] for _ in range(len(x))]
    for t, (i, j) in enumerate(zip(x, y)):
        for i2, j2 in zip(i, j):
            tul[t].append(round((i2 + j2) / 2, 3))

    return tul


def treenimine(m):
    for model in m:
        kfold = StratifiedKFold(n_splits=2, shuffle=True)
        cv_results = cross_val_score(
            model,
            preprocessing.scale(X_train, with_mean=False),
            np.ravel(Y_train),
            cv=kfold,
            scoring="accuracy",
            n_jobs=-1,
        )

        return cv_results


models = [
    LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="liblinear",
        tol=1e-10,
        multi_class="auto",
        max_iter=1500,
    )
]

dataset, tekstid = andmed()

for ngrammi_tahis in range(2):
    for ngrammi_tuup in ["word", "char", "char_wb"]:
        print(ngrammi_tuup, ngrammi_tahis)
        lopp_tulemused = []
        for o in range(2):
            tul1 = []
            algus = datetime.now()
            for ngrammi_suurus in range(1, 8):
                tul2 = []
                for df_suurus in range(0, 13, 2):
                    if ngrammi_tahis == 0:
                        xvektorid = CountVectorizer(
                            ngram_range=(
                                ngrammi_suurus,
                                ngrammi_suurus,
                            ),
                            min_df=df_suurus,
                            analyzer=ngrammi_tuup,
                        )
                    if ngrammi_tahis == 1:
                        xvektorid = CountVectorizer(
                            ngram_range=(
                                0,
                                ngrammi_suurus,
                            ),
                            min_df=df_suurus,
                            analyzer=ngrammi_tuup,
                        )

                    try:
                        xvektorid.fit(tekstid)
                        vektorid = xvektorid.transform(tekstid)
                        vektorid.toarray()
                    except ValueError:
                        tul2.append(float("nan"))
                        continue

                    keeled = dataset["emakeel"].to_numpy()

                    X_train, X_validation, Y_train, Y_validation = train_test_split(
                        vektorid, keeled, test_size=0.20, shuffle=True
                    )

                    tul2.append(round(treenimine(models).mean(), 3))
                tul1.append(tul2)
            lopp_tulemused.append(tul1)

        toodeldud_andmed = andmete_tootlus(jar=lopp_tulemused)
        pyplot.figure(figsize=(8, 8))

        try:
            pyplot.imshow(
                toodeldud_andmed,
                interpolation="nearest",
                norm=MidpointNormalize(vmin=0.2, midpoint=0.85),
            )
            pyplot.xlabel("N-gramm")
            pyplot.ylabel("Muutuja df_size")
            pyplot.colorbar()
            pyplot.xticks(np.arange(7), [1, 2, 3, 4, 5, 6, 7])
            pyplot.yticks(np.arange(7), [0, 2, 4, 6, 8, 10, 12])
            pyplot.title(
                str(f"N-grammi tüüp: {ngrammi_tahis}, N-grammi tüüp {ngrammi_tuup}")
            )
            pyplot.draw()
            # pyplot.savefig(
            #     nimi(
            #         rf"C:\Users\rasmu\OneDrive\Töölaud\Programmid\Python 3\Uurimistöö\Graafikud\N-grammi tüüp {ngrammi_tuup} kombinatsioon(0n_1y) {ngrammi_tahis}",
            #         "png",
            #     ),
            #     bbox_inches="tight",
            #     dpi=100,
            # )
            # pyplot.close()
        except TypeError:
            print(
                ngrammi_tahis,
                ngrammi_tuup,
                "#######################################",
                toodeldud_andmed,
            )

        lõpp = datetime.now()
        aeg = lõpp - algus
pyplot.show()
