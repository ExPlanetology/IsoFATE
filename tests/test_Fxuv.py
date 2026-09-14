import numpy as np

from isofate.constants import s2yr
from isofate.isofunks import Fxuv

F0 = 1e2       # main-sequence XUV flux [W/m2]
T0 = 1e6       # start time [yr]
T_SAT = 5e8    # saturation time [yr]
BETA = -1.23
F_FINAL = 10.0
T_PMS = 1e7    # pre-main-sequence phase duration [yr]
PMS_FACTOR = 1e2


def _seconds(years):
    return years / s2yr


def test_fxuv_pre_main_sequence_branch():
    time_yr = 1e6  # within (0, T_PMS)
    t = _seconds(time_yr)
    F_pms0 = F0 * PMS_FACTOR
    s = (np.log10(F0) - np.log10(F_pms0)) / (np.log10(T_PMS) - np.log10(T0))
    expected = F_pms0 * (time_yr / T0) ** s
    result = Fxuv(t, F0, t0=T0, t_sat=T_SAT, beta=BETA, step_fn=False,
                  F_final=F_FINAL, t_pms=T_PMS, pms_factor=PMS_FACTOR)
    assert np.isclose(result, expected)


def test_fxuv_saturated_branch():
    time_yr = 1e7  # below T_SAT; t_pms=0 disables the pre-main-sequence branch
    t = _seconds(time_yr)
    result = Fxuv(t, F0, t0=T0, t_sat=T_SAT, beta=BETA, step_fn=False,
                  F_final=F_FINAL, t_pms=0, pms_factor=PMS_FACTOR)
    assert result == F0


def test_fxuv_power_law_decay_branch():
    time_yr = 1e9  # past T_SAT
    t = _seconds(time_yr)
    result = Fxuv(t, F0, t0=T0, t_sat=T_SAT, beta=BETA, step_fn=False,
                  F_final=F_FINAL, t_pms=0, pms_factor=PMS_FACTOR)
    expected = F0 * (time_yr / T_SAT) ** BETA
    assert np.isclose(result, expected)


def test_fxuv_step_function_branch():
    time_yr = 1e9  # past T_SAT
    t = _seconds(time_yr)
    result = Fxuv(t, F0, t0=T0, t_sat=T_SAT, beta=BETA, step_fn=True,
                  F_final=F_FINAL, t_pms=0, pms_factor=PMS_FACTOR)
    assert result == F_FINAL
