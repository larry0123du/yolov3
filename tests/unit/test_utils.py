import pytest, unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import torch
from utils.utils import *


########################## tests for xyxy2xywh ##########################
def test_xyxy2xywh_np_type_check():
    # GIVEN
    xyxy = np.mat("0 0 0 0")

    # WHEN
    xywh = xyxy2xywh(xyxy)

    # THEN
    assert isinstance(xywh, np.ndarray)


def test_xyxy2xywh_torch_type_check():
    # GIVEN
    xyxy = torch.from_numpy(np.mat("0 0 0 0"))

    # WHEN
    xywh = xyxy2xywh(xyxy)

    # THEN
    assert isinstance(xywh, torch.Tensor)


@pytest.mark.parametrize(
    "input_type, output_type",
    [
        (np.dtype(int), np.dtype(int)),
        (np.dtype('uint'), np.dtype('uint')),
        (np.dtype(float), np.dtype(float)),
    ]
)
def test_xyxy2xywh_np_multi_dtypes(input_type, output_type):
    # GIVEN
    xyxy = np.mat("0 0 0 0", dtype=input_type)

    # WHEN
    xywh = xyxy2xywh(xyxy)

    # THEN
    assert xywh.dtype == output_type


def test_xyxy2xywh_np_zero_check():
    # GIVEN
    xyxy = np.mat("0 0 0 0", dtype=np.float32)

    # WHEN
    xywh = xyxy2xywh(xyxy)

    # THEN
    assert xywh.shape[0] == 1 and xywh.shape[1] == 4
    assert np.count_nonzero(xywh) == 0


def test_xyxy2xywh_np_nonzero():
    # GIVEN
    xyxy = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)

    # WHEN
    xywh = xyxy2xywh(xyxy)

    # THEN
    assert xywh.shape[0] == 1 and xywh.shape[1] == 4
    target = np.array([[0.2, 0.3, 0.2, 0.2]], dtype = np.float32)
    assert np.allclose(xywh, target)
