use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_linalg::Norm;
use std::collections::HashSet;

fn median(array: &Array1<f64>) -> f64 {
    let mut sorted_array = array.to_vec();
    sorted_array.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = sorted_array.len();
    if len % 2 == 0 {
        (sorted_array[len / 2 - 1] + sorted_array[len / 2]) / 2.0
    } else {
        sorted_array[len / 2]
    }
}

// Function to compute sigma
#[pyfunction]
fn compute_sigma(embeddings: PyReadonlyArray2<f64>) -> PyResult<f64> {
    let embeddings = embeddings.as_array();
    let pairwise_sq_dists = pairwise_squared_distances(&embeddings);
    let filtered_dists: Vec<f64> = pairwise_sq_dists.iter().filter(|&&x| x > 0.0).copied().collect();
    if filtered_dists.is_empty() {
        Ok(1e-10)
    } else {
        let percentile_index = (filtered_dists.len() as f64 * 0.75).ceil() as usize - 1;
        let percentile_value = *filtered_dists.get(percentile_index).unwrap_or(&1e-10);
        let sigma = (0.5 * percentile_value).sqrt();
        Ok(sigma.max(1e-10))
    }
}

// Function to compute pairwise squared distances
fn pairwise_squared_distances(embeddings: &ArrayView2<f64>) -> Array2<f64> {
    let n = embeddings.shape()[0];
    let mut dists = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let diff = &embeddings.row(i) - &embeddings.row(j);
            dists[[i, j]] = diff.dot(&diff);
        }
    }
    dists
}

// Function to compute hybrid kernel
#[pyfunction]
fn hybrid_kernel(py: Python<'_>, embeddings: PyReadonlyArray2<f64>, sigma: f64) -> PyResult<Py<PyArray2<f64>>> {
    let embeddings = embeddings.as_array();
    let norms: Array1<f64> = embeddings.map_axis(Axis(1), |row| row.norm_l2());
    let normalized_embeddings = &embeddings / &norms.insert_axis(Axis(1));
    let cosine_similarity = normalized_embeddings.dot(&normalized_embeddings.t());
    let sq_dists = pairwise_squared_distances(&embeddings);
    let gaussian_kernel = sq_dists.mapv(|x| (-x / (2.0 * sigma * sigma)).exp());
    let hybrid_kernel = gaussian_kernel * cosine_similarity;
    Ok(hybrid_kernel.into_pyarray(py).to_owned().into())
}

// Function to check similarity
#[pyfunction]
fn check_similarity(_py: Python<'_>, emb: PyReadonlyArray1<f64>, embeddings: PyReadonlyArray2<f64>, sigma: Option<f64>) -> PyResult<bool> {
    let emb = emb.as_array();
    let embeddings = embeddings.as_array();
    let sigma = match sigma {
        Some(val) => val,
        None => {
            let pairwise_dists: Array1<f64> = embeddings
                .rows()
                .into_iter()
                .map(|row| (&row - &emb).norm_l2())
                .collect();
            median(&pairwise_dists)
        }
    };

    let dists: Array1<f64> = embeddings.rows().into_iter().map(|row| (&row - &emb).norm_l2()).collect();
    let gaussian_similarity: Array1<f64> = dists.mapv(|x| (-x * x / (2.0 * sigma * sigma)).exp());
    let cosine_similarity: Array1<f64> = embeddings
        .rows()
        .into_iter()
        .map(|row| row.dot(&emb) / (row.norm_l2() * emb.norm_l2()))
        .collect();

    let combined_similarity = 0.6 * gaussian_similarity + 0.4 * cosine_similarity;
    let mean_similarity = combined_similarity.mean().unwrap_or(0.0);
    let std_similarity = combined_similarity.std(0.0);
    let threshold = mean_similarity + 1.7 * std_similarity;

    Ok(combined_similarity.iter().any(|&x| x > threshold))
}

// Function to find most similar embeddings with similarity scores
#[pyfunction]
fn find_most_similar_with_scores(embeddings: PyReadonlyArray2<f64>, reference_embeddings: PyReadonlyArray2<f64>, sigma: Option<f64>) -> PyResult<(Vec<(usize, usize)>, Vec<f64>)> {
    let embeddings = embeddings.as_array();
    let reference_embeddings = reference_embeddings.as_array();
    let sigma = match sigma {
        Some(val) => val,
        None => {
            let pairwise_dists: Array1<f64> = reference_embeddings
                .rows()
                .into_iter()
                .map(|row| (&row - &embeddings.row(0)).norm_l2())
                .collect();
            median(&pairwise_dists)
        }
    };

    let mut result = Vec::new();
    let mut similarity_scores = Vec::new();
    let mut used_indices = HashSet::new();

    for (i, emb) in embeddings.rows().into_iter().enumerate() {
        let mut best_similarity = -1.0;
        let mut best_match_index = None;

        for (j, ref_emb) in reference_embeddings.rows().into_iter().enumerate() {
            if used_indices.contains(&j) {
                continue;
            }

            let dist = (&ref_emb - &emb).norm_l2();
            let gaussian_similarity = (-dist * dist / (2.0 * sigma * sigma)).exp();
            let cosine_similarity = ref_emb.dot(&emb) / (ref_emb.norm_l2() * emb.norm_l2());

            let combined_similarity = 0.6 * gaussian_similarity + 0.4 * cosine_similarity;
            
            if combined_similarity > best_similarity {
                best_similarity = combined_similarity;
                best_match_index = Some(j);
            }
        }

        if let Some(best_match_index) = best_match_index {
            result.push((i, best_match_index));
            similarity_scores.push(best_similarity);
            used_indices.insert(best_match_index);
        }
    }

    Ok((result, similarity_scores))
}

/// A Python module implemented in Rust.
#[pymodule]
fn scorpion(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sigma, m)?)?;
    m.add_function(wrap_pyfunction!(hybrid_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(check_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(find_most_similar_with_scores, m)?)?;
    Ok(())
}
