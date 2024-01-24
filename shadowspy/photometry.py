import numpy as np


def mmpf_mh_boyd2017lpsc(phase, emission, incidence):
    """
    Use given photometric function to compute flux between npairs of sources
    (where bounce occurs) and targets (define direction)
    :param phase: ndarray (npairs)
    :param emission: ndarray (npairs)
    :param incidence: array (npairs)
    :return: ndarray (npairs)
    """

    a_0 = -2.649
    a_1 = -0.013
    a_2 = -0.274
    a_3 = 0.965

    cosinc = np.cos(incidence)
    cosemi = np.cos(emission)

    # just these 2 terms are the photom computation
    photom_num = np.exp(a_0 + a_1 * phase + a_2 * cosemi + a_3 * cosinc)
    photom_den = np.exp(a_0 + (a_2 + a_3) * cosemi)

    return photom_num / photom_den
