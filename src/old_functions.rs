// when atom positions where store in Vec<(f64, f64, f64)>

fn get_atom_coordinates(pdb: &PDB) -> Vec<(f64, f64, f64)> {
  let atoms = pdb.atoms()
    .map(|atom| {
      let pos = atom.pos();
      (pos.0, pos.1, pos.2)
    }).collect();
  atoms
}

fn get_alpha_carbon_coordinates(pdb: &PDB) -> Vec<(f64, f64, f64)> {
  let all_alpha_carbon = pdb.find(
    pdbtbx::Search::Single(
      pdbtbx::Term::AtomName("CA".to_owned())
    )
  );
  let carbons: Vec<(f64, f64, f64)>  = all_alpha_carbon.map(
    |ca| { ca.atom().pos() }
  ).collect();
  carbons
}

fn tm_score(pdb1: &PDB, pdb2: &PDB) -> f64 {
  // 1/L x (1 à L) ∑(1 + di²/d0²)
  // d0 = 1.24 √^3(L − 15) - 1.8
  // di = distance between the ith CA
  let L = pdb1.residue_count() as f64;
  let d0 = 1.24 * f64::cbrt(L-15.0) - 1.8;
  let pdb1_ca_coord = get_alpha_carbon_coordinates(pdb1);
  let pdb2_ca_coord = get_alpha_carbon_coordinates(pdb2);
  let tm_score_sum: f64 = pdb1_ca_coord.iter()
    .zip(pdb2_ca_coord.iter()) // Zip the two iterators
    .map(|((x1, y1, z1), (x2, y2, z2))| {
        let dx = x1 - x2;
        let dy = y1 - y2;
        let dz = z1 - z2;
    let quotient = (dx * dx + dy * dy + dz * dz) / d0.powi(2);
      1.0/(1.0 + quotient)
    })
    .sum();
  tm_score_sum / L
}

fn rmsd(pdb1: &PDB, pdb2: &PDB) -> f64 {
  let pdb1_coord = get_atom_coordinates(pdb1);
  let pdb2_coord = get_atom_coordinates(pdb2);
  let rmsd_sum: f64 = pdb1_coord.par_iter() // Use parallel iterator
    .zip(pdb2_coord.par_iter()) // Zip the two iterators
    .map(|((x1, y1, z1), (x2, y2, z2))| {
        let dx = x1 - x2;
        let dy = y1 - y2;
        let dz = z1 - z2;
        dx * dx + dy * dy + dz * dz
    })
    .sum(); // Sum the results in parallel
  (rmsd_sum / pdb1_coord.len() as f64).sqrt()
}
