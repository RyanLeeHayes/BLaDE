// TorchANI potential, eemlp-begin
#include <cuda_runtime.h>
#include <math.h>
#include <algorithm>

#include "main/defines.h"
#include "system/system.h"
#include "system/state.h"
#include "run/run.h"
#include "system/potential.h"
#include "domdec/domdec.h"
#include "main/real3.h"
#include "mlp/mlp.h"

#ifdef WITH_TORCH
#include <cstdio>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ========================= PBC REIMAGE =========================
template <bool flagBox, typename box_type>
__global__ void reimage_ml_coords_inplace_kernel(
    int                  nml,
    const int*  __restrict__ ml_index,
    const real3* __restrict__ position,
    float*       __restrict__ ml_coords,   // [3*nml]
    box_type               box
)
{
    if (nml <= 0) return;

    const int   ref_atom = ml_index[0];
    const real3 xref     = position[ref_atom];

    int t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < nml) {
        int   a  = ml_index[t];
        real3 xa = position[a];

        real3 dr = real3_subpbc<flagBox>(xa, xref, box);

        real x = xref.x + dr.x;
        real y = xref.y + dr.y;
        real z = xref.z + dr.z;

        ml_coords[3*t + 0] = static_cast<float>(x);
        ml_coords[3*t + 1] = static_cast<float>(y);
        ml_coords[3*t + 2] = static_cast<float>(z);
    }
}

template <bool flagBox, typename box_type>
static inline void reimage_ml_coordsT(
    Potential *p, State *s, Run *r,
    int nml,
    box_type box
)
{
    if (nml <= 0) return;

    int block = BLBO;
    int grid  = (nml + block - 1) / block;

    reimage_ml_coords_inplace_kernel<flagBox><<<grid, block, 0, r->mlpotStream>>>(
        nml,
        p->mlp_d.mlatomidx,
        (const real3*)s->position_fd,
        p->mlp_d.ml_qm_coords_s_d,
        box
    );
}

static inline void reimage_ml_coords(System *system, int nml)
{
    Potential *p = system->potential;
    State     *s = system->state;
    Run       *r = system->run;

    if (nml <= 0) return;

    if (s->typeBox) {
        reimage_ml_coordsT<true>(p, s, r, nml, s->tricBox_f);
    } else {
        reimage_ml_coordsT<false>(p, s, r, nml, s->orthBox_f);
    }
}
// ========================= REIMAGE: end =========================


// ========================= SCATTER GRADIENT/ENERGY =========================
// Input gradient is dE/dR in Hartree/Angstrom.
// It is converted here to engine units (kcal/mol/Angstrom) before atomicAdd.
// Energy input is Hartree and is converted here to engine units (kcal/mol).
__global__ void fill_grad_energy_tani_kernel(
    int                  nml,
    const int*   __restrict__ tani_ml_index,
    const float* __restrict__ tani_mlgrad_local,   // GRADIENT dE/dR, Hartree/Ang
    const float* __restrict__ tani_energy_local,   // Energy, Hartree
    real3_f*     __restrict__ force,               // engine force-like array
    real_e*      __restrict__ energy               // engine energy slot
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    real lEnergy = 0;
    extern __shared__ real sEnergy[];

    if (i < nml) {
        int atom = tani_ml_index[i];

        real_f gx = static_cast<real_f>(tani_mlgrad_local[3*i + 0] * TORCH_F_HARTREE_TO_KCALMOL);
        real_f gy = static_cast<real_f>(tani_mlgrad_local[3*i + 1] * TORCH_F_HARTREE_TO_KCALMOL);
        real_f gz = static_cast<real_f>(tani_mlgrad_local[3*i + 2] * TORCH_F_HARTREE_TO_KCALMOL);

        atomicAdd(&force[atom].x, gx);
        atomicAdd(&force[atom].y, gy);
        atomicAdd(&force[atom].z, gz);
    }

    if (energy) {
        if (i == 0) {
            lEnergy = static_cast<real>(tani_energy_local[0] * TORCH_E_HARTREE_TO_KCALMOL);
        }
        real_sum_reduce(lEnergy, sEnergy, energy);
    }
}
// ========================= SCATTER GRADIENT/ENERGY: end =========================
#endif


// ========================= TORCH MAIN =========================
void gettorchforce_tani(System *system, bool calcEnergy)
{
    Potential *p = system->potential;
    State     *s = system->state;
    Run       *r = system->run;

    if (r->calcTermFlag[eemlp] == false) return;
    if (p->atomCount <= 0) return;
    if (p->MLPModelCount <= 0) return;
    if (p->mlp_h[0].is_tani == -1) return;

#ifdef WITH_TORCH
    const int    nml    = p->mlp_h[0].ptnml;
    const size_t threeQ = 3 * static_cast<size_t>(nml);

    if (nml <= 0) return;

    cudaStream_t stream = r->mlpotStream;

    // ---- 1. Zero per-step buffers ----
    cudaMemsetAsync(p->mlp_d.ml_qm_coords_s_d, 0, threeQ * sizeof(float), stream);
    cudaMemsetAsync(p->mlp_d.ml_qm_grad_s_d,   0, threeQ * sizeof(float), stream);
    cudaMemsetAsync(p->mlp_d.ml_energy_s_d,    0, sizeof(float),          stream);
    if (cudaPeekAtLastError() != cudaSuccess) return;

    // ---- 2. Build/reimage QM coords directly from engine positions ----
    reimage_ml_coords(system, nml);
    if (cudaPeekAtLastError() != cudaSuccess) return;

    // Coordinate buffer must be complete before Torch wraps/uses it
    cudaStreamSynchronize(stream);

    // ---- 3. Put Torch work on the SAME CUDA stream ----
    at::cuda::CUDAStream torch_stream =
        at::cuda::getStreamFromExternal(stream, p->mlp_h[0].ptgpuid);
    at::cuda::CUDAStreamGuard guard(torch_stream);

    auto f32_cuda = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA);
    auto i32_cuda = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA);
    auto i64_cuda = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);

    at::Tensor ml_coords = torch::from_blob(
        p->mlp_d.ml_qm_coords_s_d,
        {1, static_cast<long>(nml), 3},
        f32_cuda
    ).clone().set_requires_grad(true);

    at::Tensor atom_types_i32 = torch::from_blob(
        p->mlp_d.mlZidx,
        {static_cast<long>(nml)},
        i32_cuda
    );

    at::Tensor atom_types = atom_types_i32.to(i64_cuda)
                                         .view({1, static_cast<long>(nml)})
                                         .contiguous();

    // ---- 4. Forward ----
    auto species_coords = c10::ivalue::Tuple::create(
        std::vector<c10::IValue>{atom_types, ml_coords}
    );

    std::vector<c10::IValue> inputs;
    inputs.reserve(1);
    inputs.emplace_back(species_coords);

    c10::IValue out_iv;
    try {
        out_iv = p->mlp_h[0].model.forward(inputs);
    } catch (const c10::Error &e) {
        std::fprintf(stderr, "TANI-DBG> model.forward FAILED: %s\n", e.what());
        std::fflush(stderr);
        return;
    } catch (...) {
        std::fprintf(stderr, "TANI-DBG> model.forward FAILED: unknown exception\n");
        std::fflush(stderr);
        return;
    }

    if (!out_iv.isTuple()) {
        std::fprintf(stderr, "TANI-DBG> model output is not tuple-like\n");
        std::fflush(stderr);
        return;
    }

    auto out_tuple = out_iv.toTuple();
    const auto& elems = out_tuple->elements();

    if (elems.size() < 2) {
        std::fprintf(stderr, "TANI-DBG> model output tuple has size < 2\n");
        std::fflush(stderr);
        return;
    }

    at::Tensor eT = elems[1].toTensor().contiguous();
    if (!eT.defined()) {
        std::fprintf(stderr, "TANI-DBG> energy tensor is undefined\n");
        std::fflush(stderr);
        return;
    }

    // ---- 5. Autograd ----
    at::Tensor e_scalar = eT.sum();

    std::vector<at::Tensor> grads;
    try {
        grads = torch::autograd::grad(
            {e_scalar},
            {ml_coords},
            {},
            false,
            false,
            false
        );
    } catch (const c10::Error &e) {
        std::fprintf(stderr, "TANI-DBG> autograd FAILED: %s\n", e.what());
        std::fflush(stderr);
        return;
    } catch (...) {
        std::fprintf(stderr, "TANI-DBG> autograd FAILED: unknown exception\n");
        std::fflush(stderr);
        return;
    }

    if (grads.empty() || !grads[0].defined()) {
        std::fprintf(stderr, "TANI-DBG> autograd returned empty/undefined gradient\n");
        std::fflush(stderr);
        return;
    }

    at::Tensor gradT     = grads[0].contiguous();
    at::Tensor grad_flat = gradT.view({-1});
    at::Tensor e1        = eT.view({-1});

    if (grad_flat.scalar_type() != at::kFloat) grad_flat = grad_flat.to(at::kFloat);
    if (e1.scalar_type()        != at::kFloat) e1        = e1.to(at::kFloat);

    // Torch work complete before any host debug reads / copy-back
    cudaStreamSynchronize(stream);

    // ---- 6. Copy Torch outputs to engine-side device buffers ----
    cudaMemcpyAsync(
        p->mlp_d.ml_qm_grad_s_d,
        grad_flat.data_ptr<float>(),
        threeQ * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream
    );
    if (cudaPeekAtLastError() != cudaSuccess) return;

    if (calcEnergy) {
        cudaMemcpyAsync(
            p->mlp_d.ml_energy_s_d,
            e1.data_ptr<float>(),
            sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        );
        if (cudaPeekAtLastError() != cudaSuccess) return;
    }

    // Engine-side buffers must be ready before scatter reads them
    cudaStreamSynchronize(stream);

    // ---- 7. Scatter gradient and energy into engine arrays ----
    {
        int block = BLBO;
        int grid  = (nml + block - 1) / block;

        int    shMem    = 0;
        real_e *pEnergy = nullptr;

        if (calcEnergy) {
            shMem   = BLBO * sizeof(real) / 32;
            pEnergy = s->energy_d + eemlp;
        }

        fill_grad_energy_tani_kernel<<<grid, block, shMem, stream>>>(
            nml,
            p->mlp_d.mlatomidx,
            p->mlp_d.ml_qm_grad_s_d,
            p->mlp_d.ml_energy_s_d,
            (real3_f*)s->force_d,
            pEnergy
        );
        if (cudaPeekAtLastError() != cudaSuccess) return;
    }

    // Final sync so engine energy is complete before printing
    cudaStreamSynchronize(stream);
#endif
}
// ========================= TORCH MAIN: end =========================
// TorchANI potential, eemlp-end