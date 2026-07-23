use std::{
    collections::HashMap,
    io::{self, ErrorKind},
    time::Instant,
};

use indicatif::{ParallelProgressIterator, ProgressIterator};
use pdbtbx::{Element, PDB};
use rayon::prelude::*;

use crate::alignment::structure_file_path;
use crate::progress::default_progress_style;

#[derive(Clone, Copy)]
struct GridAtom {
    chain_index: usize,
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct GridCell {
    x: i64,
    y: i64,
    z: i64,
}

#[derive(Clone, Copy)]
struct BoundingBox {
    minimum: [f64; 3],
    maximum: [f64; 3],
    has_atoms: bool,
}

impl BoundingBox {
    fn new() -> Self {
        Self {
            minimum: [f64::INFINITY; 3],
            maximum: [f64::NEG_INFINITY; 3],
            has_atoms: false,
        }
    }

    fn include(&mut self, x: f64, y: f64, z: f64) {
        if !x.is_finite() || !y.is_finite() || !z.is_finite() {
            return;
        }

        self.minimum[0] = self.minimum[0].min(x);
        self.minimum[1] = self.minimum[1].min(y);
        self.minimum[2] = self.minimum[2].min(z);

        self.maximum[0] = self.maximum[0].max(x);
        self.maximum[1] = self.maximum[1].max(y);
        self.maximum[2] = self.maximum[2].max(z);

        self.has_atoms = true;
    }

    fn squared_distance_to(&self, other: &Self) -> f64 {
        let mut squared_distance = 0.0;

        for axis in 0..3 {
            let gap = if self.maximum[axis] < other.minimum[axis] {
                other.minimum[axis] - self.maximum[axis]
            } else if other.maximum[axis] < self.minimum[axis] {
                self.minimum[axis] - other.maximum[axis]
            } else {
                0.0
            };

            squared_distance += gap * gap;
        }

        squared_distance
    }
}

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<u8>,
    component_count: usize,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
            rank: vec![0; size],
            component_count: size,
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        let mut root = node;

        while self.parent[root] != root {
            root = self.parent[root];
        }

        while self.parent[node] != node {
            let parent = self.parent[node];
            self.parent[node] = root;
            node = parent;
        }

        root
    }

    fn union(&mut self, left: usize, right: usize) -> bool {
        let mut left_root = self.find(left);
        let mut right_root = self.find(right);

        if left_root == right_root {
            return false;
        }

        if self.rank[left_root] < self.rank[right_root] {
            std::mem::swap(&mut left_root, &mut right_root);
        }

        self.parent[right_root] = left_root;

        if self.rank[left_root] == self.rank[right_root] {
            self.rank[left_root] += 1;
        }

        self.component_count -= 1;
        true
    }

    fn is_connected(&self) -> bool {
        self.component_count <= 1
    }
}

fn is_eligible_atom(atom: &pdbtbx::Atom) -> bool {
    atom.element() != Some(&Element::H)
}

fn collect_chain_bounding_boxes(pdb: &PDB) -> Vec<BoundingBox> {
    pdb.chains()
        .map(|chain| {
            let mut bounding_box = BoundingBox::new();

            for residue in chain.residues() {
                for atom in residue.atoms() {
                    if !is_eligible_atom(atom) {
                        continue;
                    }

                    let (x, y, z) = atom.pos();
                    bounding_box.include(x, y, z);
                }
            }

            bounding_box
        })
        .collect()
}

/// Conservatively tests connectivity using chain bounding boxes.
///
/// A disconnected result is definitive. A connected result only means that an
/// exact atom-level check is required.
fn bounding_box_graph_is_connected(
    bounding_boxes: &[BoundingBox],
    squared_cutoff: f64,
) -> bool {
    let chain_count = bounding_boxes.len();

    if chain_count <= 1 {
        return true;
    }

    let mut disjoint_set = DisjointSet::new(chain_count);

    for left in 0..chain_count {
        if !bounding_boxes[left].has_atoms {
            continue;
        }

        for right in (left + 1)..chain_count {
            if !bounding_boxes[right].has_atoms {
                continue;
            }

            let squared_distance =
                bounding_boxes[left].squared_distance_to(
                    &bounding_boxes[right],
                );

            if squared_distance <= squared_cutoff {
                disjoint_set.union(left, right);

                if disjoint_set.is_connected() {
                    return true;
                }
            }
        }
    }

    disjoint_set.is_connected()
}

fn grid_cell(x: f64, y: f64, z: f64, inverse_cell_size: f64) -> GridCell {
    GridCell {
        x: (x * inverse_cell_size).floor() as i64,
        y: (y * inverse_cell_size).floor() as i64,
        z: (z * inverse_cell_size).floor() as i64,
    }
}

fn atoms_are_in_contact(
    left: &GridAtom,
    right_x: f64,
    right_y: f64,
    right_z: f64,
    squared_cutoff: f64,
) -> bool {
    let dx = left.x - right_x;
    let dy = left.y - right_y;
    let dz = left.z - right_z;

    dx * dx + dy * dy + dz * dz <= squared_cutoff
}

/// Determine whether all chains belong to one contact-connected component.
///
/// Two chains are considered adjacent when any pair of their non-hydrogen
/// atoms is separated by at most `contact_cutoff`.
///
/// This function does not calculate or retain subgroup membership.
pub fn structure_is_fully_adjacent(
    pdb: &PDB,
    contact_cutoff: f64,
) -> io::Result<bool> {
    if !contact_cutoff.is_finite() || contact_cutoff <= 0.0 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "The contact cutoff must be a positive finite number",
        ));
    }

    let bounding_boxes = collect_chain_bounding_boxes(pdb);
    let chain_count = bounding_boxes.len();

    if chain_count == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "The structure contains no chains",
        ));
    }

    if chain_count == 1 {
        return Ok(true);
    }

    let squared_cutoff = contact_cutoff * contact_cutoff;

    // This is a conservative screening step. If the bounding-box graph is
    // disconnected, the real atom-contact graph must also be disconnected.
    if !bounding_box_graph_is_connected(
        &bounding_boxes,
        squared_cutoff,
    ) {
        return Ok(false);
    }

    let inverse_cell_size = 1.0 / contact_cutoff;
    let mut grid: HashMap<GridCell, Vec<GridAtom>> = HashMap::new();
    let mut disjoint_set = DisjointSet::new(chain_count);

    for (chain_index, chain) in pdb.chains().enumerate() {
        for residue in chain.residues() {
            for atom in residue.atoms() {
                if !is_eligible_atom(atom) {
                    continue;
                }

                let (x, y, z) = atom.pos();

                if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                    continue;
                }

                let current_cell =
                    grid_cell(x, y, z, inverse_cell_size);

                for offset_x in -1_i64..=1 {
                    for offset_y in -1_i64..=1 {
                        for offset_z in -1_i64..=1 {
                            let neighboring_cell = GridCell {
                                x: current_cell.x + offset_x,
                                y: current_cell.y + offset_y,
                                z: current_cell.z + offset_z,
                            };

                            let Some(existing_atoms) =
                                grid.get(&neighboring_cell)
                            else {
                                continue;
                            };

                            for existing_atom in existing_atoms {
                                if existing_atom.chain_index == chain_index {
                                    continue;
                                }

                                if disjoint_set.find(
                                    existing_atom.chain_index,
                                ) == disjoint_set.find(chain_index)
                                {
                                    continue;
                                }

                                if !atoms_are_in_contact(
                                    existing_atom,
                                    x,
                                    y,
                                    z,
                                    squared_cutoff,
                                ) {
                                    continue;
                                }

                                disjoint_set.union(
                                    existing_atom.chain_index,
                                    chain_index,
                                );

                                // Connectivity can only increase, so this is
                                // a definitive early exit.
                                if disjoint_set.is_connected() {
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }

                grid.entry(current_cell)
                    .or_default()
                    .push(GridAtom {
                        chain_index,
                        x,
                        y,
                        z,
                    });
            }
        }
    }

    Ok(disjoint_set.is_connected())
}

fn check_structure_file(
    structure_path: &str,
    contact_cutoff: f64,
) -> io::Result<bool> {
    let (pdb, _errors) = pdbtbx::open(structure_path).map_err(|error| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "Failed to open structure '{structure_path}': {error:?}"
            ),
        )
    })?;

    structure_is_fully_adjacent(&pdb, contact_cutoff)
}

/// Check structures sequentially.
///
/// Results have the same order as `filenames`.
pub fn all_adjacency(
    filenames: &[String],
    source_path: &str,
    contact_cutoff: f64,
) -> io::Result<Vec<bool>> {
    println!("Computing chain adjacency on a single thread....");

    let start = Instant::now();
    let style = default_progress_style();

    let results: io::Result<Vec<bool>> = filenames
        .iter()
        .progress_with_style(style)
        .map(|file_name| {
            let input_path =
                structure_file_path(source_path, file_name);

            check_structure_file(&input_path, contact_cutoff)
        })
        .collect();

    println!("Took {:?}\n", start.elapsed());
    results
}

/// Check structures in parallel, with one structure per Rayon worker.
///
/// Results have the same order as `filenames`.
pub fn parallel_all_adjacency(
    filenames: &[String],
    source_path: &str,
    contact_cutoff: f64,
) -> io::Result<Vec<bool>> {
    println!("Computing chain adjacency in parallel....");

    let start = Instant::now();
    let style = default_progress_style();

    let results: Vec<io::Result<bool>> = filenames
        .par_iter()
        .progress_with_style(style)
        .map(|file_name| {
            let input_path =
                structure_file_path(source_path, file_name);

            check_structure_file(&input_path, contact_cutoff)
        })
        .collect();

    let results: io::Result<Vec<bool>> =
        results.into_iter().collect();

    println!("Took {:?}\n", start.elapsed());
    results
}
