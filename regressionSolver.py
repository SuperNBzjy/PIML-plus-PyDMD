"""Regression workflows for physics-informed turbulence modeling.

This module retains the original functionality of the notebook-exported
``regressionSolver.py`` script while making it executable as a regular Python
module.  The implementation focuses on Step 4 of the PIML algorithm, where a
regression model is trained to map flow features to Reynolds stress
discrepancies.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Optional IPython support ----------------------------------------------------
_IPYTHON_SPEC = importlib.util.find_spec("IPython")
if _IPYTHON_SPEC is not None:
    from IPython import get_ipython as _get_ipython  # type: ignore
    from IPython.display import Image  # type: ignore
else:  # pragma: no cover - exercised only when IPython is missing
    def _get_ipython():  # type: ignore
        return None

    def Image(*_args, **_kwargs):  # type: ignore
        return None

_ipy_shell = _get_ipython()
if _ipy_shell is not None:
    _ipy_shell.run_line_magic("matplotlib", "inline")
else:  # pragma: no cover - covered implicitly when running tests
    matplotlib.use("Agg")

# Optional Keras support ------------------------------------------------------
_KERAS_SPEC = importlib.util.find_spec("keras")
if _KERAS_SPEC is not None:
    from keras.layers import Dense  # type: ignore
    from keras.models import Sequential  # type: ignore
else:  # pragma: no cover - executed when Keras is unavailable
    Dense = None  # type: ignore
    Sequential = None  # type: ignore

# Paths -----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATABASE_ROOT = REPO_ROOT / "database"
FIGURES_ROOT = REPO_ROOT / "figs"


def _load_matrix(path: Path) -> np.ndarray:
    """Load a whitespace separated matrix stored in ``path``."""
    return np.loadtxt(path)


def loadTrainingData(caseName: str, ReNum: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load training features and responses for the specified flow case."""
    base = DATABASE_ROOT / caseName
    trainFeaturesFile = base / "markers" / ReNum / "markerFile"
    trainResponsesFile = base / "deltaFields" / ReNum / "deltaField"
    return _load_matrix(trainFeaturesFile), _load_matrix(trainResponsesFile)


def loadTestData(caseName: str, ReNum: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load test features and responses for the specified flow case."""
    base = DATABASE_ROOT / caseName
    testFeaturesFile = base / "markers" / ReNum / "markerFile"
    testResponsesFile = base / "deltaFields" / ReNum / "deltaField"
    return _load_matrix(testFeaturesFile), _load_matrix(testResponsesFile)


def randomForest(
    trainFeatures: np.ndarray,
    trainResponses: np.ndarray,
    testFeatures: np.ndarray,
    maxFeatures: int | str = "log2",
    nTree: int = 100,
) -> np.ndarray:
    """Train a Random Forest regressor and return predictions for the test set."""
    regModel = RandomForestRegressor(n_estimators=nTree, max_features=maxFeatures)
    regModel.fit(trainFeatures, trainResponses)
    return regModel.predict(testFeatures)
def keras_nn(
    trainFeatures: np.ndarray,
    trainResponses: np.ndarray,
    testFeatures: np.ndarray,
    Nepochs: int = 100,
) -> np.ndarray:
    """Train a feed-forward neural network using Keras.
    Raises
    ------
    RuntimeError
        If the optional Keras dependency is not available.
    """
     if Sequential is None or Dense is None:  # pragma: no cover - requires keras
        raise RuntimeError(
            "Keras is not available. Install tensorflow/keras to use the neural "
            "network workflow."
        )
    model = Sequential()
    model.add(Dense(64, input_dim=trainFeatures.shape[1], activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="tanh"))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(trainFeatures, trainResponses, epochs=Nepochs, batch_size=200, verbose=0)
    return model.predict(testFeatures)

def plotXiEta(
    XiEta_RANS: np.ndarray,
    testResponses: np.ndarray,
    testResponsesPred: np.ndarray,
    name: str,
    symbol: str = "r^",
    *,
    interval: int = 2,
) -> None:
    """Plot Reynolds stress anisotropy in the barycentric triangle."""



    XiEta_DNS = XiEta_RANS + testResponses
    XiEta_ML = XiEta_RANS + testResponsesPred

    pointsNum = int(XiEta_RANS.shape[0])
    plt.figure()
       plt.plot([0, 1, 0.5, 0.5, 0], [0, 0, 3**0.5 / 2.0, 3**0.5 / 2.0, 0], "g-")
    p1, = plt.plot(
        XiEta_RANS[:pointsNum:interval, 0],
        XiEta_RANS[:pointsNum:interval, 1],
        "bo",
        markerfacecolor="none",
        markeredgecolor="b",
        markeredgewidth=2,
        markersize=10,
    )
    p2, = plt.plot(
        XiEta_DNS[:pointsNum:interval, 0],
        XiEta_DNS[:pointsNum:interval, 1],
        "ks",
        markerfacecolor="none",
        markeredgecolor="k",
        markeredgewidth=2,
        markersize=10,
    )
    p3, = plt.plot(
        XiEta_ML[:pointsNum:interval, 0],
        XiEta_ML[:pointsNum:interval, 1],
        symbol,
        markerfacecolor="none",
        markeredgecolor="r",
        markeredgewidth=2,
        markersize=10,
    )
    lg = plt.legend([p1, p2, p3], ["RANS", "DNS", name], loc=0)
    lg.draw_frame(False)
    plt.ylim([0, 3**0.5 / 2.0])




def comparePlotRFNN(XiEta_RANS, testResponses, testResponsesPred_RF, testResponsesPred_NN):
    
    XiEta_DNS = XiEta_RANS + testResponses
    XiEta_RF = XiEta_RANS + testResponsesPred_RF
    XiEta_NN = XiEta_RANS + testResponsesPred_NN
    # Plot Reynolds stress anisotropy in Barycentric triangle
    interval = 2
    pointsNum = int(XiEta_RANS.shape[0])
    plt.figure()
    plt.plot([0,1,0.5,0.5,0],[0,0,3**0.5/2.0,3**0.5/2.0,0],'g-')
    p1, = plt.plot(XiEta_RANS[:pointsNum:interval,0],XiEta_RANS[:pointsNum:interval,1],
                   'bo', markerfacecolor='none', markeredgecolor='b',
                   markeredgewidth=1.5, markersize=8)
    p2, = plt.plot(XiEta_DNS[:pointsNum:interval,0],XiEta_DNS[:pointsNum:interval,1],
                   'ks', markerfacecolor='none', markeredgecolor='k',
                   markeredgewidth=1.5, markersize=8)
    p3, = plt.plot(XiEta_RF[:pointsNum:interval,0],XiEta_RF[:pointsNum:interval,1],
                   'r^', markerfacecolor='none', markeredgecolor='r',
                   markeredgewidth=1.5, markersize=8)
    p4, = plt.plot(XiEta_NN[:pointsNum:interval,0],XiEta_NN[:pointsNum:interval,1],
                   'r+', markerfacecolor='none', markeredgecolor='g',
                   markeredgewidth=1.5, markersize=8)
    lg = plt.legend([p1,p2,p3, p4], ['RANS', 'DNS', 'RF', 'NN'], loc = 0)
    lg.draw_frame(False)
    plt.ylim([0,3**0.5/2.0])
    plt.show()


def iterateLines(
    dataFolderRANS: Path | str,
    testResponses: np.ndarray,
    testResponsesPred: np.ndarray,
    name: str,
    symbol: str = "r^",
    lines: Iterable[int] = (3, 5),
) -> None:
    """Iterate over sampling lines and plot barycentric coordinates."""
    dataFolder = Path(dataFolderRANS)
    indexList = [0, 98, 191, 287, 385, 483, 581, 679, 777, 875, 971]
    for iterN in lines:
        XiEta = _load_matrix(dataFolder / f"line{iterN}_XiEta.xy")
        startIndex = indexList[iterN - 1]
        endIndex = indexList[iterN]
        
    plotXiEta(
            XiEta,
            testResponses[startIndex:endIndex, :],
            testResponsesPred[startIndex:endIndex, :],
            name,
            symbol=symbol,
        )


def compareResults(
    dataFolderRANS: Path | str,
    testResponses: np.ndarray,
    testResponsesPred_RF: np.ndarray,
    testResponsesPred_NN: np.ndarray,
    lines: Iterable[int] = (3, 5),
) -> None:
    """Compare Random Forest and Neural Network predictions for the specified lines."""

    dataFolder = Path(dataFolderRANS)
    indexList = [0, 98, 191, 287, 385, 483, 581, 679, 777, 875, 971]
    for iterN in lines:
        XiEta = _load_matrix(dataFolder / f"line{iterN}_XiEta.xy")
        startIndex = indexList[iterN - 1]
        endIndex = indexList[iterN]
        comparePlotRFNN(
            XiEta,
            testResponses[startIndex:endIndex, :],
            testResponsesPred_RF[startIndex:endIndex, :],
            testResponsesPred_NN[startIndex:endIndex, :],
        )


def show_algorithm_diagrams() -> None:
    """Display algorithm diagrams when running inside IPython."""

    if Image is None:  # pragma: no cover - requires IPython
        return
        Image(filename=str(FIGURES_ROOT / "PIML-algorithm.png"))
        Image(filename=str(FIGURES_ROOT / "features.png"))

def main() -> None:
    """Execute the default regression workflow used in the tutorials."""

    show_algorithm_diagrams()
    trainFeatures, trainResponses = loadTrainingData("pehill", "Re5600")
    testFeatures, testResponses = loadTestData("pehill", "Re10595")

    time_begin_RF = time.time()
    testResponsesPred_RF = randomForest(
        trainFeatures, trainResponses, testFeatures, maxFeatures=6, nTree=100
    )
    time_end_RF = time.time()
     dataFolderRANS = DATABASE_ROOT / "pehill" / "XiEta-RANS" / "Re10595"
    iterateLines(dataFolderRANS, testResponses, testResponsesPred_RF, name="RF")
    plt.show()

    Nepochs = 1000
    testResponsesPred_NN = None
    time_begin_NN = None
    time_end_NN = None

    if Sequential is not None and Dense is not None:
        time_begin_NN = time.time()
        testResponsesPred_NN = keras_nn(
            trainFeatures, trainResponses, testFeatures, Nepochs
        )
        time_end_NN = time.time()

        iterateLines(
            dataFolderRANS,
            testResponses,
            testResponsesPred_NN,
            name="NN",
            symbol="m+",
        )
        plt.show()

        compareResults(
            dataFolderRANS, testResponses, testResponsesPred_RF, testResponsesPred_NN
        )
        plt.show()

    cost_time_RF = time_end_RF - time_begin_RF
    print(f"Random Forest runtime: {cost_time_RF:.3f} s")

    if testResponsesPred_NN is not None and time_begin_NN is not None:
        cost_time_NN = time_end_NN - time_begin_NN  # type: ignore[arg-type]
        xlabel = np.arange(2)
        plt.bar(xlabel, [cost_time_RF, cost_time_NN], 0.4)
        plt.ylabel("CPU time (sec)")
        plt.xticks(xlabel, ("RF", "NN"))
        plt.title(f"Epoches = {Nepochs}")
        plt.show()
    else:
        print(
            "Keras/TensorFlow not available - skipping neural network workflow "
            "and runtime comparison."
        )


if __name__ == "__main__":
    main()
