use half::{bf16, f16};
use memmap2::MmapOptions;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
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

/// Return the highest Shannon entropy found in any chunk of the input.
///
/// The final chunk may be shorter than `chunk_size`. Empty input and a zero
/// chunk size return 0.0 rather than attempting an invalid partition.
#[pyfunction]
pub fn shannon_entropy_chunked(data: &[u8], chunk_size: usize) -> f64 {
    if data.is_empty() || chunk_size == 0 {
        return 0.0;
    }

    data.chunks(chunk_size)
        .map(shannon_entropy_bytes)
        .fold(0.0, f64::max)
}

fn mantissa_bit_plane_entropy_for(data: &[u8], bit: u32) -> f64 {
    let float_count = data.len() / 4;
    if float_count == 0 {
        return 0.0;
    }

    let mut packed_bits = vec![0u8; float_count.div_ceil(8)];
    for (index, chunk) in data.chunks_exact(4).enumerate() {
        let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let mantissa_bit = ((word & 0x007F_FFFF) >> bit) & 1;
        packed_bits[index / 8] |= (mantissa_bit as u8) << (index % 8);
    }

    shannon_entropy_bytes(&packed_bits)
}

/// Compute entropy for the two least-significant float32 mantissa bit planes.
///
/// Input is interpreted as little-endian float32 words. The returned tuple is
/// `(bit_0_entropy, bit_1_entropy)`, with each plane packed into bytes before
/// calculating Shannon entropy. Trailing bytes that do not form a full word
/// are ignored.
#[pyfunction]
pub fn mantissa_bit_plane_entropy(data: &[u8]) -> (f64, f64) {
    (
        mantissa_bit_plane_entropy_for(data, 0),
        mantissa_bit_plane_entropy_for(data, 1),
    )
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenfordStats {
    pub mad: f64,
    pub chi_square: f64,
    pub valid_count: usize,
}

impl BenfordStats {
    fn classification(&self) -> &'static str {
        if self.mad < 0.015 {
            "normal"
        } else if self.mad < 0.035 {
            "questionable"
        } else {
            "anomalous"
        }
    }
}

/// Compute Benford's Law Mean Absolute Deviation (MAD) for leading digits of float32 values.
///
/// Benford's distribution for first non-zero digit d in {1..9}:
/// P(d) = log10(1 + 1/d)
#[pyfunction]
pub fn benford_law_mad(floats: Vec<f32>) -> f64 {
    calculate_benford_mad_slice(&floats)
}

/// Compute Benford's Law MAD and chi-square statistic together for a float slice.
#[pyfunction]
pub fn benford_law_stats(floats: Vec<f32>) -> (f64, f64) {
    let stats = calculate_benford_stats(&floats);
    (stats.mad, stats.chi_square)
}

fn calculate_benford_mad_slice(floats: &[f32]) -> f64 {
    calculate_benford_stats(floats).mad
}

fn calculate_benford_stats(floats: &[f32]) -> BenfordStats {
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
        return BenfordStats {
            mad: 0.0,
            chi_square: 0.0,
            valid_count,
        };
    }

    let mut sum_absolute_deviations = 0.0;
    let mut chi_square = 0.0;

    for d in 1..=9 {
        let expected_p = (1.0 + 1.0 / (d as f64)).log10();
        let observed_p = digit_counts[d] as f64 / valid_count as f64;
        let expected_count = expected_p * valid_count as f64;

        sum_absolute_deviations += (observed_p - expected_p).abs();
        if expected_count > 0.0 {
            chi_square += ((digit_counts[d] as f64 - expected_count).powi(2)) / expected_count;
        }
    }

    BenfordStats {
        mad: sum_absolute_deviations / 9.0,
        chi_square,
        valid_count,
    }
}

fn decode_f32_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_f16_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

fn decode_bf16_bytes(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
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
#[pyo3(signature = (path, entropy_threshold = 7.92, benford_mad_threshold = 0.035))]
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

            let float_values = match tensor_view.dtype() {
                Dtype::F32 => Some(decode_f32_bytes(data_bytes)),
                Dtype::F16 => Some(decode_f16_bytes(data_bytes)),
                Dtype::BF16 => Some(decode_bf16_bytes(data_bytes)),
                _ => None,
            };

            let benford_mad = float_values
                .as_ref()
                .map(|values| calculate_benford_mad_slice(values))
                .unwrap_or(0.0);

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
    m.add_function(wrap_pyfunction!(shannon_entropy_chunked, m)?)?;
    m.add_function(wrap_pyfunction!(mantissa_bit_plane_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(benford_law_mad, m)?)?;
    m.add_function(wrap_pyfunction!(benford_law_stats, m)?)?;
    m.add_function(wrap_pyfunction!(scan_safetensors, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_entropy_for_identical_bytes() {
        let zeros = vec![0u8; 1000];
        assert_eq!(shannon_entropy_bytes(&zeros), 0.0);

        let ones = vec![0xFFu8; 500];
        assert_eq!(shannon_entropy_bytes(&ones), 0.0);
    }

    #[test]
    fn test_high_entropy_for_uniform_distribution() {
        // Construct exact uniform distribution of all 256 byte values
        let mut uniform = Vec::new();
        for _ in 0..100 {
            for b in 0..=255u8 {
                uniform.push(b);
            }
        }
        let entropy = shannon_entropy_bytes(&uniform);
        // Shannon entropy of uniform 256 values is exactly 8.0 bits
        assert!((entropy - 8.0).abs() < 1e-4, "Entropy was {}", entropy);
    }

    #[test]
    fn test_chunked_entropy_detects_localized_payload() {
        let mut data = vec![0u8; 1024];
        data.extend((0..=255u8).cycle().take(256));
        data.extend(vec![0u8; 1024]);

        assert_eq!(shannon_entropy_chunked(&data, 256), 8.0);
        assert_eq!(shannon_entropy_chunked(&data, 0), 0.0);
        assert_eq!(shannon_entropy_chunked(&[], 1024), 0.0);
    }

    #[test]
    fn test_mantissa_bit_plane_entropy_ignores_sign_and_exponent() {
        let values = [1.0f32, -1.0, 2.0, -2.0, 4.0, -4.0, 8.0, -8.0];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();

        assert_eq!(mantissa_bit_plane_entropy(&bytes), (0.0, 0.0));
        assert_eq!(mantissa_bit_plane_entropy(&[0u8; 3]), (0.0, 0.0));
    }

    #[test]
    fn test_extract_leading_digit() {
        assert_eq!(extract_leading_digit(1.234), 1);
        assert_eq!(extract_leading_digit(9.99), 9);
        assert_eq!(extract_leading_digit(45.67), 4);
        assert_eq!(extract_leading_digit(0.00345), 3);
        assert_eq!(extract_leading_digit(0.789), 7);
    }

    #[test]
    fn test_benford_stats_include_chi_squared_and_threshold_classification() {
        let samples: Vec<f32> = (1..=5000)
            .map(|idx| ((idx % 9) + 1) as f32 + (idx as f32 * 0.01))
            .collect();

        let (mad, chi_square) = benford_law_stats(samples);
        assert!(mad >= 0.0);
        assert!(chi_square >= 0.0);
        assert!(mad > 0.0 || chi_square > 0.0);
    }

    #[test]
    fn test_benford_half_precision_support() {
        let mut values = Vec::new();
        let total_samples = 10000;

        for d in 1..=9 {
            let p_d = (1.0 + 1.0 / (d as f64)).log10();
            let count = (p_d * total_samples as f64).round() as usize;
            for i in 0..count {
                values.push(d as f32 + (i as f32 / count as f32) * 0.9);
            }
        }

        let f16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_bits().to_le_bytes())
            .collect();
        let bf16_bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| bf16::from_f32(*value).to_bits().to_le_bytes())
            .collect();

        let f16_decoded = decode_f16_bytes(&f16_bytes);
        let bf16_decoded = decode_bf16_bytes(&bf16_bytes);

        assert!(calculate_benford_mad_slice(&f16_decoded) < 0.005);
        assert!(calculate_benford_mad_slice(&bf16_decoded) < 0.005);
    }

    #[test]
    fn test_benford_mad_low_for_ideal_distribution() {
        // Generate synthetic numbers distributed according to Benford's Law
        let mut benford_samples = Vec::new();
        let total_samples = 10000;

        for d in 1..=9 {
            let p_d = (1.0 + 1.0 / (d as f64)).log10();
            let count = (p_d * total_samples as f64).round() as usize;
            for i in 0..count {
                benford_samples.push(d as f32 + (i as f32 / count as f32) * 0.9);
            }
        }

        let mad = calculate_benford_mad_slice(&benford_samples);
        // Synthetic Benford distribution should have very low MAD (< 0.005)
        assert!(mad < 0.005, "Expected MAD < 0.005, got {}", mad);
    }

    #[test]
    fn test_benford_mad_high_for_uniform_digits() {
        // Uniform digit distribution (e.g. synthetic/encrypted data)
        let mut uniform_digits = Vec::new();
        for _ in 0..1000 {
            for d in 1..=9 {
                uniform_digits.push(d as f32 + 0.5);
            }
        }

        let mad = calculate_benford_mad_slice(&uniform_digits);
        // Non-Benford distribution exhibits significant MAD (> 0.03)
        assert!(mad > 0.03, "Expected MAD > 0.03, got {}", mad);
    }
}
