from unittest.mock import patch

import arrow
import numpy as np
import pytest

from dynawind import __version__
from dynawind import dynawind as dw


def test_version():
    assert __version__ == "0.4.0"


@pytest.mark.parametrize("data_length", [1, 600, 600000])
@patch("dynawind.dynawind.processSignal")
def test_signal_class(mock_process_signal, data_length):
    """Very basic tests

    Assert that ``source`` equals ``site`` after initialization,
    and that calculated statistics are floats
    """
    site = "Belwind"
    group = "acceleration"
    fs = 100
    data = np.random.rand(data_length)
    name = "TP_ACC_LAT19_315deg_X"
    unit = "g"
    timestamp = arrow.utcnow()

    with patch("dynawind.dynawind.get_site", return_value=site):
        signal = dw.Signal(site, group, fs, data, name, unit, timestamp)
        mock_process_signal.assert_called_once()
        assert signal.source == site
        assert isinstance(signal.rms(), float)
        assert isinstance(signal.median(), float)
        assert isinstance(signal.mean(), float)
        assert isinstance(signal.std(), float)


def test_get_site_known():
    assert dw.get_site("BBC01") == "Belwind"


def test_get_site_unknown():
    assert dw.get_site("BBZ01") != "Belwind"
    assert dw.get_site("BBZ01") == "unknown"


@pytest.mark.parametrize("site,locations", [("Belwind", ["BBC01"]), ("C-Power", ["CPA7", "CPG2"])])
def test_get_locations(site, locations):
    assert dw.get_locations(site) == locations
