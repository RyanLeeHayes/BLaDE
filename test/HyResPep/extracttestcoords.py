#! /usr/bin/env python

import MDAnalysis as mda

u = mda.Universe("demo.psf","peptide.dcd")

for ts in u.trajectory:
    frame_idx = u.trajectory.frame
    output_filename = f'testcoords/frame_{frame_idx:04d}.crd'
    u.atoms.write(output_filename)
