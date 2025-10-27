import numpy as np
import pytest

import pybinding as pb
from pybinding.repository import graphene


@pytest.mark.parametrize("matrix_format", ["CSR", "ELL"])
def test_cpu_gpu_ldos_match(matrix_format):
    kernel = pb.lorentz_kernel()
    cpu_model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(4))
    cpu_kpm = pb.kpm(cpu_model, kernel=kernel, matrix_format=matrix_format,
                     optimal_size=True, interleaved=True, silent=True)

    try:
        gpu_model = pb.Model(graphene.monolayer(), graphene.hexagon_ac(4))
        gpu_kpm = pb.kpm_cuda(gpu_model, kernel=kernel, matrix_format=matrix_format,
                              optimal_size=True, interleaved=True, silent=True)
    except Exception as exc:
        pytest.skip(f"GPU backend unavailable: {exc}")

    energy = np.linspace(-0.5, 0.5, 8)
    broadening = 0.1
    position = [0, 0]

    cpu_ldos = cpu_kpm.calc_ldos(energy, broadening, position)
    gpu_ldos = gpu_kpm.calc_ldos(energy, broadening, position)

    assert np.allclose(gpu_ldos.data, cpu_ldos.data, rtol=1e-5, atol=1e-7)
