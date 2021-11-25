import pytest
import create_simulated_image
import utils.phase_map as pm

def test_init_pm_with_invalid_mode():
    with pytest.raises(ValueError, match="Mode 0 is not a recognised mode."):
        pm.phase_map(0, mode= '0')