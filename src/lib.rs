mod alignment;
mod chain_distances;
pub mod cli;
mod contacts;
mod interface;
mod metrics;
mod progress;
mod scoring;
mod structure_clustering;

pub use alignment::{all_alignment, parallel_all_alignment, structure_files_from_directory};
pub use chain_distances::{
    all_min_distances, filter_chain_pairs, minimal_chain_distances, sanitize_data, ChainDistance,
};
pub use contacts::{
    all_clashes, all_contacts, all_contacts_with_clashes, clashes_threshold, count_clashes,
    results_to_csv_string, write_interface_contacts_csv, FlatContactRecord, ModelContactsSummary,
};
pub use dockq_rs::{
    all_dockq, all_interfaces_contacts, all_interfaces_contacts_from_files, compare_contacts,
    compare_contacts_from_files, compute_dockq, compute_dockq_from_files, extract_contacts,
    extract_contacts_from_files, DockQAlignedContact, DockQBatchResult, DockQConfig, DockQContact,
    DockQContacts, DockQError, DockQInterfaceContactsResult, DockQMode, DockQNativeContact,
    DockQPartners, DockQResult, ResidueKey,
};
pub use interface::{all_iplddt, compute_interface_plddt};
pub use metrics::all_distances;
pub use scoring::{all_scores_computation, score_interface};
pub use structure_clustering::{
    assign_clusters, cluster_structure_files, cluster_structures, compute_reduced_points,
    ClusterAssignment, ReducedPoint,
};

#[cfg(feature = "python")]
mod python;
