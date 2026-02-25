use std::collections::HashMap;
use std::time::Instant;

use indicatif::{ParallelProgressIterator};
use pdbtbx::Atom;
use pdbtbx::ContainsAtomConformer;
use pdbtbx::ContainsAtomConformerResidueChain;
use rayon::prelude::*;
use rstar::{PointDistance, RTree, RTreeObject, AABB};

use crate::progress::default_progress_style;

/// Minimal distance between two chains
#[derive(Clone, Debug)]
pub struct ChainDistance {
    pub chain1: String,
    pub chain2: String,
    pub min_distance: f64,
}

/// An atom
#[derive(Clone)]
struct AtomWrapper {
    atom: Atom,
}

impl RTreeObject for AtomWrapper {
    type Envelope = AABB<[f64; 3]>;

    fn envelope(&self) -> Self::Envelope {
        let pos = self.atom.pos();
        AABB::from_point([pos.0, pos.1, pos.2])
    }
}

impl PointDistance for AtomWrapper {
    /// Compute distance between two point in space (atoms)
    fn distance_2(&self, point: &[f64; 3]) -> f64 {
        let pos = self.atom.pos();
        let dx = pos.0 - point[0];
        let dy = pos.1 - point[1];
        let dz = pos.2 - point[2];
        dx * dx + dy * dy + dz * dz
    }

    fn contains_point(&self, point: &[f64; 3]) -> bool {
        self.distance_2(point) == 0.0
    }
}

/// Compute the minimal distance between two chains
pub fn minimal_chain_distances(pdb_file: &str) -> Vec<ChainDistance> {
    let (pdb, _errors) = pdbtbx::open(pdb_file).expect("Failed to open the PDB file");

    let mut chain_atoms: HashMap<String, Vec<Atom>> = HashMap::new();
    for atom_with_hierarchy in pdb.atoms_with_hierarchy() {
        let chain_id = atom_with_hierarchy.chain().id().to_string();
        chain_atoms
            .entry(chain_id)
            .or_insert_with(Vec::new)
            .push(atom_with_hierarchy.atom().to_owned());
    }

    let chain_ids: Vec<_> = chain_atoms.keys().cloned().collect();
    let mut results = Vec::new();

    for i in 0..chain_ids.len() {
        for j in (i + 1)..chain_ids.len() {
            let chain_id1 = &chain_ids[i];
            let chain_id2 = &chain_ids[j];

            let atoms1 = &chain_atoms[chain_id1];
            let atoms2 = &chain_atoms[chain_id2];

            let (fast_atoms, slow_atoms) = if atoms1.len() > atoms2.len() {
                (atoms1, atoms2)
            } else {
                (atoms2, atoms1)
            };

            let wrappers: Vec<AtomWrapper> = fast_atoms
                .iter()
                .cloned()
                .map(|atom| AtomWrapper { atom })
                .collect();
            let rtree = RTree::bulk_load(wrappers);

            let mut min_distance = f64::MAX;

            for atom in slow_atoms {
                let pos = atom.pos();
                let query_point = [pos.0, pos.1, pos.2];
                if let Some(nearest) = rtree.nearest_neighbor(&query_point) {
                    let npos = nearest.atom.pos();
                    let dx = pos.0 - npos.0;
                    let dy = pos.1 - npos.1;
                    let dz = pos.2 - npos.2;
                    let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                    if distance < min_distance {
                        min_distance = distance;
                    }
                }
            }

            results.push(ChainDistance {
                chain1: chain_id1.clone(),
                chain2: chain_id2.clone(),
                min_distance,
            });
        }
    }

    results
}

/// Compute minimal distance between two chains for a set of structures in parallel
pub fn all_min_distances(input_dir: &str, pdb_file_names: &[String]) -> Vec<Vec<ChainDistance>> {
    println!(
        "Computing minimal chain distances on {} structures",
        pdb_file_names.len()
    );
    let start = Instant::now();
    let style = default_progress_style();

    let all_distances: Vec<Vec<ChainDistance>> = pdb_file_names
        .par_iter()
        .progress_with_style(style)
        .map(|filename| {
            let pdb_file = format!("{input_dir}/{filename}");
            brute_force_minimal_chain_distances(&pdb_file)
        })
        .collect();

    let duration = start.elapsed();
    println!("Took {:?}\n", duration);
    all_distances
}

fn brute_force_minimal_chain_distances(pdb_file: &str) -> Vec<ChainDistance> {
    let (pdb, _errors) = pdbtbx::open(pdb_file).expect("Failed to open the PDB file");
    let mut chain_atoms: HashMap<String, Vec<Atom>> = HashMap::new();
    for atom_with_hierarchy in pdb.atoms_with_hierarchy() {
        let chain_id = atom_with_hierarchy.chain().id().to_string();
        chain_atoms
            .entry(chain_id)
            .or_insert_with(Vec::new)
            .push(atom_with_hierarchy.atom().to_owned());
    }
    let mut results = Vec::new();
    let chain_ids: Vec<String> = chain_atoms.keys().cloned().collect();
    for i in 0..chain_ids.len() {
        for j in (i + 1)..chain_ids.len() {
            let chain_id1 = &chain_ids[i];
            let chain_id2 = &chain_ids[j];
            let atoms1 = &chain_atoms[chain_id1];
            let atoms2 = &chain_atoms[chain_id2];

            let mut min_distance = f64::MAX;
            for atom1 in atoms1 {
                for atom2 in atoms2 {
                    let distance = atom1.distance(atom2);
                    if distance < min_distance {
                        min_distance = distance;
                    }
                }
            }
            results.push(ChainDistance {
                chain1: chain_id1.clone(),
                chain2: chain_id2.clone(),
                min_distance,
            });
        }
    }
    results
}

pub fn filter_chain_pairs(distances: &[ChainDistance], pairs_str: &str) -> Vec<ChainDistance> {
    let pairs: Vec<(String, String)> = pairs_str
        .split(',')
        .filter_map(|s| {
            let trimmed = s.trim();
            if trimmed.len() == 2 {
                let mut chars = trimmed.chars();
                let first = chars.next().unwrap().to_string();
                let second = chars.next().unwrap().to_string();
                Some((first, second))
            } else {
                None
            }
        })
        .collect();

    distances
        .iter()
        .filter(|cd| {
            pairs.iter().any(|(p1, p2)| {
                (cd.chain1 == *p1 && cd.chain2 == *p2) || (cd.chain1 == *p2 && cd.chain2 == *p1)
            })
        })
        .cloned()
        .collect()
}

pub fn sanitize_data(
    file_names: &[String],
    data: &[Vec<ChainDistance>],
) -> (Vec<String>, Vec<Vec<f64>>) {
    if data.is_empty() {
        return (vec![], vec![]);
    }

    let mut unique_keys = Vec::new();
    let mut key_to_index = HashMap::new();

    for cd in &data[0] {
        let key = canonical_chain_key(&cd.chain1, &cd.chain2);
        if !key_to_index.contains_key(&key) {
            key_to_index.insert(key.clone(), unique_keys.len());
            unique_keys.push(key);
        }
    }

    let mut min_distances: Vec<Vec<f64>> =
        vec![Vec::with_capacity(file_names.len()); unique_keys.len()];
    for file_data in data {
        let mut file_map = HashMap::new();
        for cd in file_data {
            let key = canonical_chain_key(&cd.chain1, &cd.chain2);
            file_map.insert(key, cd.min_distance);
        }
        for key in &unique_keys {
            let idx = key_to_index[key];
            if let Some(&dist) = file_map.get(key) {
                min_distances[idx].push(dist);
            } else {
                min_distances[idx].push(f64::NAN);
            }
        }
    }

    (unique_keys, min_distances)
}

/// Standardize chain order by sorting them
fn canonical_chain_key(chain1: &str, chain2: &str) -> String {
    if chain1 < chain2 {
        format!("{}{}", chain1, chain2)
    } else {
        format!("{}{}", chain2, chain1)
    }
}
