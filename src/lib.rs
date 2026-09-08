use memmap2::MmapOptions;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::fs::File;

/// Result for an individual scanned tensor in a .safetensors file.
#[pyclass]
#[derive(Clone, Debug, serde::Serialize)]
pub struct TensorScanResult {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub dtype: String,
    #[pyo3(get)]
    pub shape: Vec<usize>,
    #[pyo3(get)]
    pub num_elements: usize,
    #[pyo3(get)]
    pub entropy: f64,
    #[pyo3(get)]
    pub benford_mad: f64,
    #[pyo3(get)]
    pub is_suspicious: bool,
    #[pyo3(get)]
    pub anomaly_reasons: Vec<String>,
}

#[pymethods]
impl TensorScanResult {
    fn __repr__(&self) -> String {
        format!(
            "<TensorScanResult name='{}' shape={:?} entropy={:.4} benford_mad={:.4} suspicious={}>",
            self.name, self.shape, self.entropy, self.benford_mad, self.is_suspicious
        )
    }
}

/// Calculate Shannon entropy over raw byte slices.
///
/// Returns a value between 0.0 (uniform single byte) and 8.0 (pure random/encrypted).
#[pyfunction]
pub fn shannon_entropy_bytes(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }

    let total = data.len() as f64;
    let mut entropy = 0.0;

    for &count in &counts {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    entropy
}

/// Compute Benford's Law Mean Absolute Deviation (MAD) for leading digits of float32 values.
///
/// Benford's distribution for first non-zero digit d in {1..9}:
/// P(d) = log10(1 + 1/d)
#[pyfunction]
pub fn benford_law_mad(floats: Vec<f32>) -> f64 {
    calculate_benford_mad_slice(&floats)
}

fn calculate_benford_mad_slice(floats: &[f32]) -> f64 {
    let mut digit_counts = [0usize; 10]; // index 1..=9
    let mut valid_count = 0usize;

    for &val in floats {
        if !val.is_finite() || val == 0.0 {
            continue;
        }

        let abs_val = val.abs();
        let leading_digit = extract_leading_digit(abs_val);
        if (1..=9).contains(&leading_digit) {
            digit_counts[leading_digit] += 1;
            valid_count += 1;
        }
    }

    if valid_count < 100 {
        // Insufficient sample size to reliably apply Benford's Law
        return 0.0;
    }

    // Expected Benford probabilities for d = 1..=9
    let mut sum_absolute_deviations = 0.0;
    for d in 1..=9 {
        let expected_p = (1.0 + 1.0 / (d as f64)).log10();
        let observed_p = digit_counts[d] as f64 / valid_count as f64;
        sum_absolute_deviations += (observed_p - expected_p).abs();
    }

    sum_absolute_deviations / 9.0
}

#[inline]
fn extract_leading_digit(mut val: f32) -> usize {
    if val >= 1.0 {
        while val >= 10.0 {
            val /= 10.0;
        }
        val as usize
    } else {
        while val < 1.0 && val > 0.0 {
            val *= 10.0;
        }
        val as usize
    }
}

/// Zero-copy scan of a .safetensors model file using memory mapping.
///
/// Analyzes each tensor for steganographic anomalies using Shannon Entropy and Benford's Law.
#[pyfunction]
#[pyo3(signature = (path, entropy_threshold = 7.92, benford_mad_threshold = 0.04))]
pub fn scan_safetensors(
    path: &str,
    entropy_threshold: f64,
    benford_mad_threshold: f64,
) -> PyResult<Vec<TensorScanResult>> {
    let file = File::open(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open model file '{}': {}", path, e)))?;

    let mmap = unsafe {
        MmapOptions::new()
            .map(&file)
            .map_err(|e| PyIOError::new_err(format!("Failed to memory map file '{}': {}", path, e)))?
    };

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| PyIOError::new_err(format!("Failed to parse safetensors file: {}", e)))?;

    let tensor_names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

    // Parallel analysis using Rayon
    let results: Vec<TensorScanResult> = tensor_names
        .par_iter()
        .filter_map(|name| {
            let tensor_view = tensors.tensor(name).ok()?;
            let data_bytes = tensor_view.data();
            let shape = tensor_view.shape().to_vec();
            let dtype = format!("{:?}", tensor_view.dtype());
            let num_elements = shape.iter().product();

            let entropy = shannon_entropy_bytes(data_bytes);

            // Benford's Law check on float32 tensors
            let mut benford_mad = 0.0;
            if dtype.contains("F32") && data_bytes.len() % 4 == 0 {
                let f32_vals: Vec<f32> = data_bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                benford_mad = calculate_benford_mad_slice(&f32_vals);
            }

            let mut anomaly_reasons = Vec::new();
            let mut is_suspicious = false;

            // Anomaly heuristics:
            // 1. High Shannon entropy indicates encrypted or compressed payload injected in weights.
            if entropy > entropy_threshold {
                is_suspicious = true;
                anomaly_reasons.push(format!(
                    "Unusually high Shannon entropy ({:.4} > threshold {:.4}) - possible encrypted/compressed payload",
                    entropy, entropy_threshold
                ));
            }

            // 2. Benford's Law MAD deviation indicates artificial/non-natural digit distribution.
            if benford_mad > benford_mad_threshold {
                is_suspicious = true;
                anomaly_reasons.push(format!(
                    "Significant Benford's Law MAD deviation ({:.4} > threshold {:.4}) - unnatural weight distribution",
                    benford_mad, benford_mad_threshold
                ));
            }

            Some(TensorScanResult {
                name: name.clone(),
                dtype,
                shape,
                num_elements,
                entropy,
                benford_mad,
                is_suspicious,
                anomaly_reasons,
            })
        })
        .collect();

    Ok(results)
}

/// Aegis-Tensor core native extension module.
#[pymodule]
fn aegis_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TensorScanResult>()?;
    m.add_function(wrap_pyfunction!(shannon_entropy_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(benford_law_mad, m)?)?;
    m.add_function(wrap_pyfunction!(scan_safetensors, m)?)?;
    Ok(())
}
