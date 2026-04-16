import numpy as np
import torch
import torchani
import MDAnalysis as mda


# ============================================================
# User settings
# ============================================================
PSF_FILE = "ethanol.psf"
CRD_FILE = "ethanol.crd"
OUT_PT   = "ethanol.pt"


# ============================================================
# Helpers
# ============================================================
HARTREE_TO_KCALMOL = 627.509474


mass_to_atomic_number = {
    1.0080: 1,     # H
    12.0110: 6,    # C
    14.0070: 7,    # N
    15.9994: 8,    # O
}


def get_atomic_number(mass: float, tol: float = 0.02) -> int:
    for ref_mass, atomic_num in mass_to_atomic_number.items():
        if abs(mass - ref_mass) < tol:
            return atomic_num
    raise ValueError(f"No atomic number found for mass {mass:.6f}")


def print_tensor_info(name: str, x: torch.Tensor) -> None:
    print(
        f"{name}: dtype={x.dtype}, shape={tuple(x.shape)}, "
        f"device={x.device}, requires_grad={x.requires_grad}, "
        f"contiguous={x.is_contiguous()}"
    )


def extract_energy(output) -> torch.Tensor:
    # TorchANI output is SpeciesEnergies / tuple-like: (species, energies)
    if isinstance(output, tuple):
        return output[1]
    if hasattr(output, "energies"):
        return output.energies
    raise TypeError(f"Unsupported output type: {type(output)}")


# ============================================================
# Load system
# ============================================================
print("=" * 80)
print("Loading CHARMM files")
print("=" * 80)

u = mda.Universe(PSF_FILE, CRD_FILE)
masses = u.atoms.masses
positions = u.atoms.positions
natoms = len(masses)

print(f"Loaded {natoms} atoms.")
print(f"Positions dtype from MDAnalysis: {positions.dtype}, shape: {positions.shape}")

atomic_numbers = [get_atomic_number(float(m)) for m in masses]
print("Atomic numbers:", atomic_numbers)

species = torch.tensor([atomic_numbers], dtype=torch.long)
coords = torch.tensor(np.array([positions], dtype=np.float32), dtype=torch.float32)

print_tensor_info("species", species)
print_tensor_info("coords", coords)


# ============================================================
# Build plain TorchANI model
# ============================================================
print("\n" + "=" * 80)
print("Building plain TorchANI ANI2x model")
print("=" * 80)

model = torchani.models.ANI2x(periodic_table_index=True)
model.eval()

print(f"Model type: {type(model)}")


# ============================================================
# Eager check: energy + autograd
# ============================================================
print("\n" + "=" * 80)
print("Eager check")
print("=" * 80)

coords_eager = coords.clone().requires_grad_(True)

output_eager = model((species, coords_eager))
energy_eager = extract_energy(output_eager)   # shape usually (1,)
energy_scalar_eager = energy_eager.sum()

grad_eager = torch.autograd.grad(
    outputs=energy_scalar_eager,
    inputs=coords_eager,
    grad_outputs=None,
    retain_graph=False,
    create_graph=False,
    allow_unused=False,
)[0]

print_tensor_info("energy_eager", energy_eager)
print_tensor_info("grad_eager", grad_eager)

print(f"Eager energy (Hartree): {energy_eager[0].item():.10f}")
print(f"Eager energy (kcal/mol): {energy_eager[0].item() * HARTREE_TO_KCALMOL:.10f}")
print("First 5 atoms of eager gradient [Hartree/Ang]:")
for i in range(min(natoms, 5)):
    gx, gy, gz = grad_eager[0, i].tolist()
    print(f"  atom {i:2d}: ({gx: .10e}, {gy: .10e}, {gz: .10e})")


# ============================================================
# Save TorchScript model
# ============================================================
print("\n" + "=" * 80)
print("Saving TorchScript model")
print("=" * 80)

scripted = torch.jit.script(model)
torch.jit.save(scripted, OUT_PT)

print(f"Saved TorchScript model to: {OUT_PT}")


# ============================================================
# Reload and verify: energy + autograd
# ============================================================
print("\n" + "=" * 80)
print("Reload check")
print("=" * 80)

loaded = torch.jit.load(OUT_PT)
loaded.eval()

coords_loaded = coords.clone().requires_grad_(True)

output_loaded = loaded((species, coords_loaded))
energy_loaded = extract_energy(output_loaded)
energy_scalar_loaded = energy_loaded.sum()

grad_loaded = torch.autograd.grad(
    outputs=energy_scalar_loaded,
    inputs=coords_loaded,
    grad_outputs=None,
    retain_graph=False,
    create_graph=False,
    allow_unused=False,
)[0]

print_tensor_info("energy_loaded", energy_loaded)
print_tensor_info("grad_loaded", grad_loaded)

print(f"Loaded energy (Hartree): {energy_loaded[0].item():.10f}")
print(f"Loaded energy (kcal/mol): {energy_loaded[0].item() * HARTREE_TO_KCALMOL:.10f}")
print(f"Energy difference (Hartree): {abs((energy_eager - energy_loaded).item()):.3e}")

grad_diff = (grad_eager - grad_loaded).abs().max().item()
print(f"Max gradient difference (Hartree/Ang): {grad_diff:.3e}")

print("First 5 atoms of loaded gradient [Hartree/Ang]:")
for i in range(min(natoms, 5)):
    gx, gy, gz = grad_loaded[0, i].tolist()
    print(f"  atom {i:2d}: ({gx: .10e}, {gy: .10e}, {gz: .10e})")


# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Saved file: {OUT_PT}")
print("Input convention:")
print("  - species = atomic numbers")
print("  - species dtype = torch.int64")
print("  - coordinates dtype = torch.float32")
print("  - coordinates unit = Angstrom")
print("Output convention:")
print(f"  - energy dtype = {energy_loaded.dtype}")
print(f"  - gradient dtype = {grad_loaded.dtype}")
print("  - energy unit = Hartree")
print("  - gradient unit = Hartree/Angstrom")
print("Autograd through saved .pt: OK")
