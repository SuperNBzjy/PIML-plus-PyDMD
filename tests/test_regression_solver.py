import os
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import regressionSolver as rs


@pytest.fixture(scope="module")
def data_shapes():
    trainF, trainR = rs.loadTrainingData("pehill", "Re5600")
    testF, testR = rs.loadTestData("pehill", "Re10595")
    return trainF, trainR, testF, testR


def test_loaders_work_outside_repo(tmp_path, data_shapes):
    trainF, trainR, testF, testR = data_shapes
    cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        trainF2, trainR2 = rs.loadTrainingData("pehill", "Re5600")
        testF2, testR2 = rs.loadTestData("pehill", "Re10595")
    finally:
        os.chdir(cwd)

    np.testing.assert_allclose(trainF2, trainF)
    np.testing.assert_allclose(trainR2, trainR)
    np.testing.assert_allclose(testF2, testF)
    np.testing.assert_allclose(testR2, testR)


def test_random_forest_predictions_shape(data_shapes):
    trainF, trainR, testF, testR = data_shapes
    preds = rs.randomForest(trainF, trainR, testF, maxFeatures=6, nTree=1)
    assert preds.shape == testR.shape
