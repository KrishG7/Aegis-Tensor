use half::{bf16, f16};
use memmap2::MmapOptions;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
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
    /// Bias-corrected entropy of each byte lane (index 0 = least significant byte).
    /// Empty for non-float tensors.
    #[pyo3(get)]
    pub lane_entropies: Vec<f64>,
    /// Entropy of the sign/exponent byte lane. Near 8 bits means the float values
    /// were overwritten with random-looking data.
    #[pyo3(get)]
    pub exponent_lane_entropy: Option<f64>,
    /// Lowest entropy across the two least significant mantissa bit planes. Clean
    /// trained weights are close to 8 bits; structured payloads pull this down.
    #[pyo3(get)]
    pub low_plane_entropy: Option<f64>,
    /// Fraction of float elements that are non-finite or implausibly large.
    /// Near zero for trained weights; large for overwritten or random data.
    #[pyo3(get)]
    pub out_of_range_fraction: Option<f64>,
    #[pyo3(get)]
    pub is_suspicious: bool,
    #[pyo3(get)]
    pub anomaly_reasons: Vec<String>,
    /// Stable codes, one per entry in `anomaly_reasons`: pooled_entropy, exponent_lane,
    /// out_of_range, low_plane, benford, unreadable. Callers can rely on these; the reason
    /// text is for people and may change wording.
    #[pyo3(get)]
    pub rules: Vec<String>,
    /// Tensor names contributing to a pooled small-tensor result. Empty for individual tensors.
    #[pyo3(get)]
    pub pooled_from: Vec<String>,
    /// Statistics of the values packed inside this tensor (GPTQ/AWQ int32 words, bitsandbytes
    /// 4-bit bytes, FP8). Only set when the format is known; no rule reads it yet.
    #[pyo3(get)]
    pub packed: Option<PackedStats>,
}

type AnalyzedTensor = (TensorScanResult, Option<(Dtype, Vec<u8>)>);

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
    for (index, chunk) in data.as_chunks::<4>().0.iter().enumerate() {
        let word = u32::from_le_bytes(*chunk);
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

    for (d, &count) in digit_counts.iter().enumerate().skip(1) {
        let expected_p = (1.0 + 1.0 / (d as f64)).log10();
        let observed_p = count as f64 / valid_count as f64;
        let expected_count = expected_p * valid_count as f64;

        sum_absolute_deviations += (observed_p - expected_p).abs();
        if expected_count > 0.0 {
            chi_square += ((count as f64 - expected_count).powi(2)) / expected_count;
        }
    }

    BenfordStats {
        mad: sum_absolute_deviations / 9.0,
        chi_square,
        valid_count,
    }
}

fn decode_f32_bytes(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect()
}

fn decode_f16_bytes(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
        .collect()
}

fn decode_bf16_bytes(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| bf16::from_bits(u16::from_le_bytes(*chunk)).to_f32())
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

/// Minimum tensor size (bytes) for the pooled whole-tensor entropy rule.
const MIN_BLOB_BYTES: usize = 4096;
/// Minimum element count for the per-lane rule. Bias correction keeps clean small
/// tensors near 2.7 bits in the sign/exponent lane, so this can stay low.
const MIN_LANE_ELEMENTS: usize = 256;
/// Minimum element count for the low-bit-plane rule. The packed plane must hold
/// at least 4096 bytes, otherwise the entropy estimate is too noisy to use.
const MIN_PLANE_ELEMENTS: usize = 32_768;
/// Magnitude above which a trained weight is implausible. Random bit patterns
/// land above this about 45% of the time, while trained weights almost never do.
const MAX_PLAUSIBLE_MAGNITUDE: f32 = 1.0e4;
/// Fraction of out-of-range elements (non-finite or above the magnitude cap) that
/// marks a tensor as overwritten. Clean tensors measured zero in the sweep.
const OUT_OF_RANGE_FRACTION: f64 = 0.001;
/// Minimum number of valid leading digits before Benford's MAD is reported.
const MIN_BENFORD_SAMPLES: usize = 5_000;

/// Miller-Madow bias-corrected Shannon entropy of a byte stream, in bits.
///
/// Plug-in entropy underestimates small samples, so a short random blob would
/// otherwise look less random than it is. The result is capped at 8 bits.
fn corrected_entropy(symbols: &[u8]) -> f64 {
    let n = symbols.len();
    if n == 0 {
        return 0.0;
    }

    let mut counts = [0usize; 256];
    for &s in symbols {
        counts[s as usize] += 1;
    }

    let total = n as f64;
    let mut entropy = 0.0;
    let mut observed = 0usize;
    for &count in &counts {
        if count > 0 {
            observed += 1;
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
    }

    let corrected = entropy + (observed as f64 - 1.0) / (2.0 * total * std::f64::consts::LN_2);
    corrected.min(8.0)
}

/// Bytes at position `lane` of every `width`-byte element (little-endian layout).
fn byte_lane(data: &[u8], width: usize, lane: usize) -> Vec<u8> {
    data.chunks_exact(width).map(|chunk| chunk[lane]).collect()
}

/// Bias-corrected entropy of bit `plane` of every element, packed into bytes.
///
/// Works on 16- and 32-bit words. Bits 0 and 1 are mantissa bits in f16, bf16
/// and f32 alike, so the same plane index means the same thing for all three.
fn plane_entropy(data: &[u8], width: usize, plane: u32) -> f64 {
    let count = data.len() / width;
    let mut packed = vec![0u8; count.div_ceil(8)];
    for (index, chunk) in data.chunks_exact(width).enumerate() {
        let word = if width == 4 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as u32
        };
        let bit = ((word >> plane) & 1) as u8;
        packed[index / 8] |= bit << (index % 8);
    }
    corrected_entropy(&packed)
}

/// Fraction of elements whose bit `plane` is set.
///
/// A constant plane carries no payload. Models stored at lower precision and
/// upcast to float32 have all-zero low bits, and so do tensors this tool disarms.
/// The low-plane rule only examines planes that are actually mixed.
fn plane_one_fraction(data: &[u8], width: usize, plane: u32) -> f64 {
    let mut ones = 0usize;
    let mut count = 0usize;
    for chunk in data.chunks_exact(width) {
        let word = if width == 4 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as u32
        };
        ones += ((word >> plane) & 1) as usize;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        ones as f64 / count as f64
    }
}

/// True when most sampled elements hold distinct values. Constant or heavily
/// quantized tensors (e.g. LayerNorm weights initialized to 1.0) have skewed low
/// bits without any payload, so the plane rule must skip them.
fn has_diverse_values(data: &[u8], width: usize) -> bool {
    const SAMPLE: usize = 1 << 20;
    let mut seen = std::collections::HashSet::new();
    let mut sampled = 0usize;
    for chunk in data.chunks_exact(width).take(SAMPLE) {
        let word = if width == 4 {
            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
        } else {
            u16::from_le_bytes([chunk[0], chunk[1]]) as u32
        };
        seen.insert(word);
        sampled += 1;
    }
    sampled > 0 && seen.len() * 2 >= sampled
}

/// Thresholds shared by every tensor rule.
#[derive(Clone, Copy)]
struct Thresholds {
    entropy: f64,
    benford: Option<f64>,
    exponent: f64,
    /// Exponent-lane threshold for float16 only. Its high byte also holds two mantissa
    /// bits, so clean float16 tensors score higher than float32 or bfloat16 ones.
    exponent_f16: f64,
    plane: f64,
}

/// The exponent-lane threshold for a dtype: float16 has its own, every other float dtype
/// uses the general one.
fn exponent_threshold_for(dtype: Dtype, t: &Thresholds) -> f64 {
    if matches!(dtype, Dtype::F16) {
        t.exponent_f16
    } else {
        t.exponent
    }
}

/// Apply every tensor rule to one tensor's raw little-endian bytes. Used for safetensors
/// files and, through `scan_tensor_bytes`, for tensors read from other formats.
fn analyze_tensor(
    name: &str,
    dtype: Dtype,
    shape: Vec<usize>,
    data_bytes: &[u8],
    t: Thresholds,
) -> TensorScanResult {
    let num_elements: usize = shape.iter().product();

    let entropy = shannon_entropy_bytes(data_bytes);

    let is_float = matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16);
    let width = match dtype {
        Dtype::F32 => 4,
        Dtype::F16 | Dtype::BF16 => 2,
        _ => 1,
    };

    let float_values = match dtype {
        Dtype::F32 => Some(decode_f32_bytes(data_bytes)),
        Dtype::F16 => Some(decode_f16_bytes(data_bytes)),
        Dtype::BF16 => Some(decode_bf16_bytes(data_bytes)),
        _ => None,
    };
    let out_of_range_fraction = float_values
        .as_ref()
        .filter(|values| !values.is_empty())
        .map(|values| {
            let bad = values
                .iter()
                .filter(|v| !v.is_finite() || v.abs() > MAX_PLAUSIBLE_MAGNITUDE)
                .count();
            bad as f64 / values.len() as f64
        });
    let benford_stats = float_values
        .as_ref()
        .map(|values| calculate_benford_stats(values));
    let benford_mad = benford_stats.map(|s| s.mad).unwrap_or(0.0);

    let lane_entropies: Vec<f64> = if is_float {
        (0..width)
            .map(|lane| corrected_entropy(&byte_lane(data_bytes, width, lane)))
            .collect()
    } else {
        Vec::new()
    };

    let exponent_lane_entropy = if is_float && num_elements >= MIN_LANE_ELEMENTS {
        lane_entropies.last().copied()
    } else {
        None
    };

    // Only mixed planes are examined. A constant plane (upcast low precision,
    // or bits zeroed by `aegis disarm`) is not a payload.
    let low_plane_entropy = if is_float
        && num_elements >= MIN_PLANE_ELEMENTS
        && has_diverse_values(data_bytes, width)
    {
        (0..2u32)
            .filter(|&plane| {
                let ones = plane_one_fraction(data_bytes, width, plane);
                ones > 0.01 && ones < 0.99
            })
            .map(|plane| plane_entropy(data_bytes, width, plane))
            .reduce(f64::min)
    } else {
        None
    };

    let mut anomaly_reasons = Vec::new();
    let mut rules: Vec<String> = Vec::new();

    // Rule 1: whole-tensor entropy (encrypted or compressed blob).
    if data_bytes.len() >= MIN_BLOB_BYTES && entropy > t.entropy {
        rules.push("pooled_entropy".to_string());
        anomaly_reasons.push(format!(
            "Unusually high Shannon entropy ({:.4} > threshold {:.4}) - possible encrypted/compressed payload",
            entropy, t.entropy
        ));
    }

    // Rule 2: sign/exponent lane is random-looking, so the float values were overwritten.
    let exponent_threshold = exponent_threshold_for(dtype, &t);
    if let Some(lane_entropy) = exponent_lane_entropy {
        if lane_entropy > exponent_threshold {
            rules.push("exponent_lane".to_string());
            anomaly_reasons.push(format!(
                "Sign/exponent bytes are near-uniform ({:.4} > threshold {:.4}) - float values overwritten with encrypted or compressed data",
                lane_entropy, exponent_threshold
            ));
        }
    }

    // Rule 2b: implausible magnitudes or non-finite values (overwritten floats).
    if let Some(fraction) = out_of_range_fraction {
        if num_elements >= MIN_LANE_ELEMENTS && fraction > OUT_OF_RANGE_FRACTION {
            rules.push("out_of_range".to_string());
            anomaly_reasons.push(format!(
                "{:.2}% of values are non-finite or above {:.0} in magnitude - float values overwritten with random data",
                fraction * 100.0,
                MAX_PLAUSIBLE_MAGNITUDE
            ));
        }
    }

    // Rule 3: low mantissa bits are structured (payload hidden in the least significant bits).
    if let Some(plane_entropy) = low_plane_entropy {
        if plane_entropy < t.plane {
            rules.push("low_plane".to_string());
            anomaly_reasons.push(format!(
                "Low mantissa bit planes are structured ({:.4} < floor {:.4}) - possible payload hidden in the least significant bits",
                plane_entropy, t.plane
            ));
        }
    }

    // Optional: Benford's Law, only with enough samples and an explicit threshold.
    if let (Some(threshold), Some(stats)) = (t.benford, benford_stats) {
        if stats.valid_count >= MIN_BENFORD_SAMPLES && benford_mad > threshold {
            rules.push("benford".to_string());
            anomaly_reasons.push(format!(
                "Benford's Law MAD deviation ({:.4} > threshold {:.4}) - non-natural digit distribution",
                benford_mad, threshold
            ));
        }
    }

    TensorScanResult {
        name: name.to_string(),
        dtype: format!("{:?}", dtype),
        shape,
        num_elements,
        entropy,
        benford_mad,
        lane_entropies,
        exponent_lane_entropy,
        low_plane_entropy,
        out_of_range_fraction,
        is_suspicious: !anomaly_reasons.is_empty(),
        rules,
        anomaly_reasons,
        pooled_from: Vec::new(),
        packed: None,
    }
}

fn analyze_small_tensor_pool(
    dtype: Dtype,
    tensors: &[(String, Vec<u8>, usize)],
    thresholds: Thresholds,
) -> Option<TensorScanResult> {
    let num_elements: usize = tensors.iter().map(|(_, _, count)| *count).sum();
    if num_elements < MIN_LANE_ELEMENTS {
        return None;
    }

    let mut data_bytes = Vec::new();
    for (_, bytes, _) in tensors {
        data_bytes.extend_from_slice(bytes);
    }
    let pooled_from = tensors.iter().map(|(name, _, _)| name.clone()).collect();
    let name = format!("pooled small {:?} tensors ({})", dtype, tensors.len());
    let mut result = analyze_tensor(&name, dtype, vec![num_elements], &data_bytes, thresholds);
    if !result.is_suspicious {
        return None;
    }
    result.pooled_from = pooled_from;
    Some(result)
}

/// Zero-copy scan of a .safetensors model file using memory mapping.
///
/// Each tensor is checked with three rules:
/// 1. Whole-tensor Shannon entropy above `entropy_threshold` (encrypted or
///    compressed blob, for tensors of at least 4 KiB).
/// 2. Sign/exponent byte lane above `exponent_entropy_threshold` (float values
///    overwritten with random-looking data). Clean weights measure about 2.7 bits,
///    and wide-dynamic-range clean weights reach about 6 bits. The default of 6.5
///    therefore detects overwrites covering roughly half of a tensor's floats.
///    float16 uses `exponent_f16_entropy_threshold` (7.2) instead: its high byte also
///    holds two mantissa bits, and clean float16 biases measured up to 6.85
///    (docs/calibration.md).
/// 3. Low mantissa bit planes below `plane_floor` (structured data hidden in the
///    least significant bits). Clean weights measure about 8 bits.
///
/// Benford's Law MAD is reported on every tensor, but it only flags a tensor when
/// `benford_mad_threshold` is given. It is noisy on small tensors and is off by default.
#[pyfunction]
#[pyo3(signature = (
    path,
    entropy_threshold = 7.92,
    benford_mad_threshold = None,
    exponent_entropy_threshold = 6.5,
    plane_floor = 7.5,
    exponent_f16_entropy_threshold = 7.2,
    packed_formats = None
))]
#[allow(clippy::too_many_arguments)]
pub fn scan_safetensors(
    path: &str,
    entropy_threshold: f64,
    benford_mad_threshold: Option<f64>,
    exponent_entropy_threshold: f64,
    plane_floor: f64,
    exponent_f16_entropy_threshold: f64,
    packed_formats: Option<HashMap<String, (String, u32)>>,
) -> PyResult<Vec<TensorScanResult>> {
    // tensor name -> packed format and bit width, as read by Python from the repo's config.
    let mut packed_kinds: HashMap<String, PackedKind> = HashMap::new();
    for (tensor, (format, bits)) in packed_formats.unwrap_or_default() {
        let kind =
            parse_packed_kind(&format, bits).map_err(pyo3::exceptions::PyValueError::new_err)?;
        packed_kinds.insert(tensor, kind);
    }
    let file = File::open(path)
        .map_err(|e| PyIOError::new_err(format!("Failed to open model file '{}': {}", path, e)))?;

    let mmap = unsafe {
        MmapOptions::new().map(&file).map_err(|e| {
            PyIOError::new_err(format!("Failed to memory map file '{}': {}", path, e))
        })?
    };

    let tensors = SafeTensors::deserialize(&mmap)
        .map_err(|e| PyIOError::new_err(format!("Failed to parse safetensors file: {}", e)))?;

    let tensor_names: Vec<String> = tensors.names().into_iter().map(|s| s.to_string()).collect();

    // Parallel analysis using Rayon
    let thresholds = Thresholds {
        entropy: entropy_threshold,
        benford: benford_mad_threshold,
        exponent: exponent_entropy_threshold,
        exponent_f16: exponent_f16_entropy_threshold,
        plane: plane_floor,
    };
    let analyzed: Vec<AnalyzedTensor> = tensor_names
        .par_iter()
        .map(|name| {
            let tensor_view = match tensors.tensor(name) {
                Ok(view) => view,
                Err(error) => {
                    // An unreadable tensor is never silently dropped: a tampered
                    // tensor is exactly the kind a scanner must not skip.
                    return (
                        TensorScanResult {
                            name: name.clone(),
                            dtype: "unknown".to_string(),
                            shape: Vec::new(),
                            num_elements: 0,
                            entropy: 0.0,
                            benford_mad: 0.0,
                            lane_entropies: Vec::new(),
                            exponent_lane_entropy: None,
                            low_plane_entropy: None,
                            out_of_range_fraction: None,
                            is_suspicious: true,
                            rules: vec!["unreadable".to_string()],
                            anomaly_reasons: vec![format!(
                                "Tensor data could not be read ({}) - possible tampering",
                                error
                            )],
                            pooled_from: Vec::new(),
                            packed: None,
                        },
                        None,
                    );
                }
            };
            let dtype = tensor_view.dtype();
            let shape = tensor_view.shape().to_vec();
            let num_elements: usize = shape.iter().product();
            let small_float_data = if num_elements < MIN_LANE_ELEMENTS
                && matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16)
            {
                Some((dtype, tensor_view.data().to_vec()))
            } else {
                None
            };
            let mut result = analyze_tensor(name, dtype, shape, tensor_view.data(), thresholds);
            let packed_kind = packed_kinds
                .get(name)
                .copied()
                .or((dtype == Dtype::F8_E4M3).then_some(PackedKind::Fp8E4M3));
            result.packed =
                packed_kind.and_then(|kind| packed_stats(kind, tensor_view.data()).ok());
            (result, small_float_data)
        })
        .collect();

    let mut results = Vec::new();
    let mut small_f32 = Vec::new();
    let mut small_f16 = Vec::new();
    let mut small_bf16 = Vec::new();
    for (name, (result, small_float_data)) in tensor_names.iter().cloned().zip(analyzed) {
        if let Some((dtype, data)) = small_float_data {
            let entry = (name, data, result.num_elements);
            match dtype {
                Dtype::F32 => small_f32.push(entry),
                Dtype::F16 => small_f16.push(entry),
                Dtype::BF16 => small_bf16.push(entry),
                _ => unreachable!("only float tensors are pooled"),
            }
        }
        results.push(result);
    }
    for pool in [&mut small_f32, &mut small_f16, &mut small_bf16] {
        pool.sort_by(|left, right| left.0.cmp(&right.0));
    }
    for (dtype, contributors) in [
        (Dtype::F32, small_f32),
        (Dtype::F16, small_f16),
        (Dtype::BF16, small_bf16),
    ] {
        if let Some(pooled_result) = analyze_small_tensor_pool(dtype, &contributors, thresholds) {
            results.push(pooled_result);
        }
    }

    Ok(results)
}

/// (dtype, bytes per element) for the dtype names used by safetensors.
fn parse_dtype(name: &str) -> Option<(Dtype, usize)> {
    Some(match name {
        "F64" => (Dtype::F64, 8),
        "F32" => (Dtype::F32, 4),
        "F16" => (Dtype::F16, 2),
        "BF16" => (Dtype::BF16, 2),
        "I64" => (Dtype::I64, 8),
        "I32" => (Dtype::I32, 4),
        "I16" => (Dtype::I16, 2),
        "I8" => (Dtype::I8, 1),
        "U64" => (Dtype::U64, 8),
        "U32" => (Dtype::U32, 4),
        "U16" => (Dtype::U16, 2),
        "U8" => (Dtype::U8, 1),
        "BOOL" => (Dtype::BOOL, 1),
        _ => return None,
    })
}

/// Apply the safetensors tensor rules to one tensor read from another format
/// (GGUF, ONNX, Keras). `data` is the tensor's raw little-endian bytes.
#[pyfunction]
#[pyo3(signature = (
    name,
    dtype,
    shape,
    data,
    entropy_threshold = 7.92,
    benford_mad_threshold = None,
    exponent_entropy_threshold = 6.5,
    plane_floor = 7.5,
    exponent_f16_entropy_threshold = 7.2
))]
#[allow(clippy::too_many_arguments)]
pub fn scan_tensor_bytes(
    name: &str,
    dtype: &str,
    shape: Vec<usize>,
    data: &[u8],
    entropy_threshold: f64,
    benford_mad_threshold: Option<f64>,
    exponent_entropy_threshold: f64,
    plane_floor: f64,
    exponent_f16_entropy_threshold: f64,
) -> PyResult<TensorScanResult> {
    let (parsed, width) = parse_dtype(dtype).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("unsupported dtype '{}'", dtype))
    })?;
    let expected = shape.iter().product::<usize>() * width;
    if expected != data.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "tensor '{}': shape {:?} of {} needs {} bytes, got {}",
            name,
            shape,
            dtype,
            expected,
            data.len()
        )));
    }
    let thresholds = Thresholds {
        entropy: entropy_threshold,
        benford: benford_mad_threshold,
        exponent: exponent_entropy_threshold,
        exponent_f16: exponent_f16_entropy_threshold,
        plane: plane_floor,
    };
    Ok(analyze_tensor(name, parsed, shape, data, thresholds))
}

// --- Packed quantized tensors ---------------------------------------------------------
//
// A quantized tensor stores several small values per larger word, so the byte rules see a
// jumble. These functions read the values out in place from the memory map, one word at a
// time, and summarise them. They report statistics only; no rule uses them yet.

/// How the values of a tensor are packed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PackedKind {
    /// GPTQ/AWQ `qweight`: little-endian int32 words holding 32 / `bits` values, lowest first.
    /// `bits` is 2, 4 or 8.
    GptqInt { bits: u32 },
    /// bitsandbytes 4-bit: one byte holds two values, the first in the high nibble.
    Bnb4,
    /// FP8 E4M3: one value per byte, laid out as sign (1) | exponent (4) | mantissa (3).
    Fp8E4M3,
}

/// Statistics of the unpacked values of one tensor.
#[pyclass]
#[derive(Clone, Debug, serde::Serialize)]
pub struct PackedStats {
    /// "gptq_int2", "gptq_int4", "gptq_int8", "bnb_4bit" or "fp8_e4m3".
    #[pyo3(get)]
    pub format: String,
    /// Bits per unpacked value.
    #[pyo3(get)]
    pub bits: u32,
    #[pyo3(get)]
    pub num_values: usize,
    /// Bias-corrected entropy of the value histogram, in bits (at most `bits`).
    #[pyo3(get)]
    pub value_entropy: f64,
    /// Bias-corrected entropy per lane. For the word formats a lane is the position of a
    /// value inside its word or byte (0 = lowest bits). For FP8 the lanes are the fields
    /// [sign, exponent, mantissa].
    #[pyo3(get)]
    pub lane_entropies: Vec<f64>,
    /// Fraction of values whose bit `i` is set, for `i` in 0..bits (0 = least significant).
    #[pyo3(get)]
    pub plane_one_fractions: Vec<f64>,
}

/// The values of one GPTQ/AWQ int32 word, lowest bits first. `bits` must be 2, 4 or 8.
pub fn unpack_gptq_word(word: u32, bits: u32) -> Vec<u8> {
    let mask = (1u32 << bits) - 1;
    (0..32 / bits)
        .map(|lane| ((word >> (lane * bits)) & mask) as u8)
        .collect()
}

/// The two 4-bit values of one bitsandbytes byte, in storage order (high nibble first).
pub fn unpack_bnb_byte(byte: u8) -> [u8; 2] {
    [byte >> 4, byte & 0x0F]
}

/// (sign, exponent, mantissa) of one FP8 E4M3 byte.
pub fn fp8_e4m3_fields(byte: u8) -> (u8, u8, u8) {
    (byte >> 7, (byte >> 3) & 0x0F, byte & 0x07)
}

/// Plug-in entropy of a histogram with the same small-sample correction as `corrected_entropy`,
/// capped at `cap` bits (the most a value of that width can carry).
fn histogram_entropy(counts: &[u64; 256], cap: f64) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    let mut entropy = 0.0;
    let mut observed = 0usize;
    for &count in counts.iter().filter(|&&c| c > 0) {
        observed += 1;
        let p = count as f64 / total;
        entropy -= p * p.log2();
    }
    let corrected = entropy + (observed as f64 - 1.0) / (2.0 * total * std::f64::consts::LN_2);
    corrected.min(cap)
}

/// Unpack `data` in place and summarise the values. Errors if the format cannot be unpacked
/// or `data` is not a whole number of packed units.
pub fn packed_stats(kind: PackedKind, data: &[u8]) -> Result<PackedStats, String> {
    let (format, bits, lane_caps, unit): (String, u32, Vec<f64>, usize) = match kind {
        PackedKind::GptqInt { bits } if matches!(bits, 2 | 4 | 8) => (
            format!("gptq_int{bits}"),
            bits,
            vec![bits as f64; (32 / bits) as usize],
            4,
        ),
        PackedKind::GptqInt { bits } => {
            return Err(format!(
                "cannot unpack {bits}-bit GPTQ words: only 2, 4 and 8 are supported"
            ))
        }
        PackedKind::Bnb4 => ("bnb_4bit".to_string(), 4, vec![4.0, 4.0], 1),
        PackedKind::Fp8E4M3 => ("fp8_e4m3".to_string(), 8, vec![1.0, 4.0, 3.0], 1),
    };
    if data.is_empty() || !data.len().is_multiple_of(unit) {
        return Err(format!(
            "{} bytes is not a whole number of {}-byte packed units",
            data.len(),
            unit
        ));
    }

    let mut values = [0u64; 256];
    let mut lanes = vec![[0u64; 256]; lane_caps.len()];
    let mut ones = vec![0u64; bits as usize];
    let mut count = 0u64;
    let mut record = |lane: usize, value: u8| {
        lanes[lane][value as usize] += 1;
        values[value as usize] += 1;
        count += 1;
        for (plane, set) in ones.iter_mut().enumerate() {
            *set += ((value >> plane) & 1) as u64;
        }
    };

    match kind {
        PackedKind::GptqInt { bits } => {
            let mask = (1u32 << bits) - 1;
            for chunk in data.chunks_exact(4) {
                let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                for lane in 0..(32 / bits) {
                    record(lane as usize, ((word >> (lane * bits)) & mask) as u8);
                }
            }
        }
        PackedKind::Bnb4 => {
            for &byte in data {
                let [first, second] = unpack_bnb_byte(byte);
                // Lane 0 is the low nibble, lane 1 the high nibble (storage order is high first).
                record(0, second);
                record(1, first);
            }
        }
        PackedKind::Fp8E4M3 => {}
    }

    // FP8 values are whole bytes whose fields are the lanes, so it is handled apart.
    if kind == PackedKind::Fp8E4M3 {
        let mut field_lanes = vec![[0u64; 256]; 3];
        let mut field_values = [0u64; 256];
        let mut field_ones = [0u64; 8];
        for &byte in data {
            let (sign, exponent, mantissa) = fp8_e4m3_fields(byte);
            field_lanes[0][sign as usize] += 1;
            field_lanes[1][exponent as usize] += 1;
            field_lanes[2][mantissa as usize] += 1;
            field_values[byte as usize] += 1;
            for (plane, set) in field_ones.iter_mut().enumerate() {
                *set += ((byte >> plane) & 1) as u64;
            }
        }
        let n = data.len();
        return Ok(PackedStats {
            format,
            bits,
            num_values: n,
            value_entropy: histogram_entropy(&field_values, 8.0),
            lane_entropies: field_lanes
                .iter()
                .zip(&lane_caps)
                .map(|(hist, cap)| histogram_entropy(hist, *cap))
                .collect(),
            plane_one_fractions: field_ones.iter().map(|&c| c as f64 / n as f64).collect(),
        });
    }

    Ok(PackedStats {
        format,
        bits,
        num_values: count as usize,
        value_entropy: histogram_entropy(&values, bits as f64),
        lane_entropies: lanes
            .iter()
            .zip(&lane_caps)
            .map(|(hist, cap)| histogram_entropy(hist, *cap))
            .collect(),
        plane_one_fractions: ones.iter().map(|&c| c as f64 / count as f64).collect(),
    })
}

/// Map a format name and bit width from Python to a `PackedKind`.
fn parse_packed_kind(format: &str, bits: u32) -> Result<PackedKind, String> {
    match format {
        "gptq" | "awq" => Ok(PackedKind::GptqInt { bits }),
        "bnb4" | "bnb_4bit" => Ok(PackedKind::Bnb4),
        "fp8_e4m3" => Ok(PackedKind::Fp8E4M3),
        other => Err(format!(
            "unknown packed format '{other}': use gptq, awq, bnb4 or fp8_e4m3"
        )),
    }
}

/// Statistics of the values packed inside one tensor's raw bytes.
///
/// `format` is "gptq", "awq" (int32 words, `bits` = 2, 4 or 8), "bnb4" or "fp8_e4m3".
#[pyfunction]
#[pyo3(signature = (format, data, bits = 4))]
pub fn unpack_stats(format: &str, data: &[u8], bits: u32) -> PyResult<PackedStats> {
    let kind = parse_packed_kind(format, bits).map_err(pyo3::exceptions::PyValueError::new_err)?;
    packed_stats(kind, data).map_err(pyo3::exceptions::PyValueError::new_err)
}

/// Aegis-Tensor core native extension module.
#[pymodule]
fn aegis_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<TensorScanResult>()?;
    m.add_class::<PackedStats>()?;
    m.add_function(wrap_pyfunction!(unpack_stats, m)?)?;
    m.add_function(wrap_pyfunction!(scan_tensor_bytes, m)?)?;
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
    fn test_small_tensor_pool_below_lane_floor_is_not_analyzed() {
        let count = MIN_LANE_ELEMENTS - 1;
        let tensors = vec![("small".to_string(), weight_like_f32(count, 7), count)];
        let thresholds = Thresholds {
            entropy: 7.92,
            benford: None,
            exponent: 6.5,
            exponent_f16: 7.2,
            plane: 7.5,
        };

        assert!(analyze_small_tensor_pool(Dtype::F32, &tensors, thresholds).is_none());
    }

    #[test]
    fn test_small_tensor_pool_detects_each_float_dtype_and_keeps_name_order() {
        let thresholds = Thresholds {
            entropy: 7.92,
            benford: None,
            exponent: 6.5,
            exponent_f16: 7.2,
            plane: 7.5,
        };
        for (dtype, width) in [(Dtype::F32, 4), (Dtype::F16, 2), (Dtype::BF16, 2)] {
            let tensors: Vec<(String, Vec<u8>, usize)> = (0..4)
                .map(|index| {
                    let name = format!("chunk.{}", index);
                    let bytes = (0..64usize)
                        .flat_map(|sample| {
                            let high_byte = (index * 64 + sample) as u8;
                            (0..width).map(
                                move |lane| {
                                    if lane + 1 == width {
                                        high_byte
                                    } else {
                                        0
                                    }
                                },
                            )
                        })
                        .collect();
                    (name, bytes, 64)
                })
                .collect();
            let result = analyze_small_tensor_pool(dtype, &tensors, thresholds)
                .expect("random-looking pool should have findings");
            assert!(result.is_suspicious);
            assert_eq!(
                result.pooled_from,
                vec![
                    "chunk.0".to_string(),
                    "chunk.1".to_string(),
                    "chunk.2".to_string(),
                    "chunk.3".to_string()
                ]
            );
        }
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

    /// Deterministic pseudo-random bytes (xorshift), so tests need no RNG crate.
    fn pseudo_random_bytes(len: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state & 0xFF) as u8
            })
            .collect()
    }

    /// Small-magnitude float32 values that look like trained weights (uniform in +-0.02).
    fn weight_like_f32(count: usize, seed: u64) -> Vec<u8> {
        pseudo_random_bytes(count * 4, seed)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| {
                let unit = u32::from_le_bytes(*c) as f64 / u32::MAX as f64;
                ((unit * 2.0 - 1.0) as f32 * 0.02).to_le_bytes()
            })
            .collect()
    }

    #[test]
    fn test_corrected_entropy_stays_within_bounds() {
        let uniform: Vec<u8> = (0..=255u8).cycle().take(256 * 50).collect();
        let h = corrected_entropy(&uniform);
        assert!(h <= 8.0 && h > 7.9, "uniform corrected entropy was {}", h);
        assert_eq!(corrected_entropy(&[]), 0.0);
    }

    #[test]
    fn test_exponent_lane_separates_weights_from_random_bytes() {
        let weights = weight_like_f32(20_000, 7);
        let weight_lane = corrected_entropy(&byte_lane(&weights, 4, 3));
        assert!(
            weight_lane < 5.0,
            "weight sign/exponent lane was {}",
            weight_lane
        );

        let blob = pseudo_random_bytes(80_000, 9);
        let blob_lane = corrected_entropy(&byte_lane(&blob, 4, 3));
        assert!(
            blob_lane > 7.5,
            "random sign/exponent lane was {}",
            blob_lane
        );
    }

    #[test]
    fn test_low_plane_is_random_for_clean_weights_and_structured_for_text() {
        let weights = weight_like_f32(100_000, 3);
        let clean_plane = plane_entropy(&weights, 4, 0);
        assert!(clean_plane > 7.9, "clean low plane was {}", clean_plane);

        let mut text_bits = weights.clone();
        let payload = b"connect 10.0.0.5:4444 /bin/sh ".repeat(2000);
        for (index, chunk) in text_bits.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let word = u32::from_le_bytes(*chunk);
            let byte = payload[(index / 8) % payload.len()];
            let bit = ((byte >> (index % 8)) & 1) as u32;
            *chunk = ((word & !1) | bit).to_le_bytes();
        }
        let structured_plane = plane_entropy(&text_bits, 4, 0);
        assert!(
            structured_plane < 7.5,
            "structured low plane was {}",
            structured_plane
        );
    }

    #[test]
    fn test_constant_tensor_is_not_diverse() {
        let ones: Vec<u8> = std::iter::repeat_n(1.0f32.to_le_bytes(), 40_000)
            .flatten()
            .collect();
        assert!(!has_diverse_values(&ones, 4));
        let weights = weight_like_f32(40_000, 5);
        assert!(has_diverse_values(&weights, 4));
    }

    #[test]
    fn test_zeroed_low_bits_are_a_constant_plane_not_a_payload() {
        // Upcast low-precision weights, or bits zeroed by disarming, have constant low planes.
        let masked: Vec<u8> = weight_like_f32(50_000, 11)
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| {
                let word = u32::from_le_bytes(*c) & !0xFFu32;
                word.to_le_bytes()
            })
            .collect();
        assert_eq!(plane_one_fraction(&masked, 4, 0), 0.0);
        assert_eq!(plane_one_fraction(&masked, 4, 1), 0.0);

        let mixed = weight_like_f32(50_000, 11);
        let fraction = plane_one_fraction(&mixed, 4, 0);
        assert!(
            fraction > 0.4 && fraction < 0.6,
            "mixed plane fraction was {}",
            fraction
        );
    }

    /// Two-byte words whose high (sign/exponent) byte cycles through 112 values: an
    /// exponent lane of about 6.8 bits, between the general and the float16 thresholds.
    fn wide_exponent_lane_words(n: usize) -> Vec<u8> {
        (0..n)
            .flat_map(|i| [(i % 251) as u8, (i % 112) as u8])
            .collect()
    }

    #[test]
    fn test_analyze_tensor_uses_the_f16_exponent_threshold_only_for_f16() {
        let thresholds = Thresholds {
            entropy: 7.92,
            benford: None,
            exponent: 6.5,
            exponent_f16: 7.2,
            plane: 7.5,
        };
        let data = wide_exponent_lane_words(4096);

        let as_f16 = analyze_tensor("b", Dtype::F16, vec![4096], &data, thresholds);
        let as_bf16 = analyze_tensor("b", Dtype::BF16, vec![4096], &data, thresholds);

        let lane = as_f16.exponent_lane_entropy.expect("lane is measured");
        assert!(lane > 6.5 && lane < 7.2, "fixture lane was {}", lane);
        assert!(!as_f16.rules.contains(&"exponent_lane".to_string()));
        assert!(as_bf16.rules.contains(&"exponent_lane".to_string()));
    }

    #[test]
    fn unpack_gptq_int4_word_gives_eight_values() {
        // Nibbles 0..=7, lowest nibble first: 0x7654_3210.
        assert_eq!(
            unpack_gptq_word(0x7654_3210, 4),
            vec![0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(unpack_gptq_word(0xFFFF_FFFF, 4), vec![15; 8]);
        // 8-bit variant: four values per word.
        assert_eq!(
            unpack_gptq_word(0x4433_2211, 8),
            vec![0x11, 0x22, 0x33, 0x44]
        );
        // 2-bit variant: sixteen values per word.
        assert_eq!(unpack_gptq_word(0b11_10_01_00, 2)[..4], [0, 1, 2, 3]);
        assert_eq!(unpack_gptq_word(0, 2).len(), 16);
    }

    #[test]
    fn unpack_bnb_byte_gives_two_values() {
        // bitsandbytes stores the first value in the high nibble.
        assert_eq!(unpack_bnb_byte(0xA3), [0xA, 0x3]);
        assert_eq!(unpack_bnb_byte(0x00), [0, 0]);
        assert_eq!(unpack_bnb_byte(0xFF), [15, 15]);
    }

    #[test]
    fn fp8_e4m3_sign_exponent_extraction() {
        // 1 1010 110: sign 1, exponent 10, mantissa 6.
        assert_eq!(fp8_e4m3_fields(0b1101_0110), (1, 0b1010, 0b110));
        assert_eq!(fp8_e4m3_fields(0x00), (0, 0, 0));
        // 0 1111 111 is the NaN encoding of E4M3.
        assert_eq!(fp8_e4m3_fields(0x7F), (0, 15, 7));
    }

    #[test]
    fn packed_stats_read_the_bytes_in_place_and_count_every_value() {
        // Two int32 words of eight nibbles each.
        let data: Vec<u8> = [0x7654_3210u32, 0x7654_3210]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let stats = packed_stats(PackedKind::GptqInt { bits: 4 }, &data).unwrap();
        assert_eq!(stats.num_values, 16);
        assert_eq!(stats.bits, 4);
        assert_eq!(stats.lane_entropies.len(), 8);
        // Every lane holds one constant value, so no lane has any entropy.
        assert!(stats.lane_entropies.iter().all(|e| *e == 0.0));
        // Eight distinct values, equally often: 3 bits, plus the small-sample correction (7/(2*16*ln 2) = 0.32).
        assert!(
            (stats.value_entropy - 3.32).abs() < 0.05,
            "{}",
            stats.value_entropy
        );
        // The top bit of a nibble is set for values 4..=7, i.e. half of them.
        assert_eq!(stats.plane_one_fractions.len(), 4);
        assert!((stats.plane_one_fractions[2] - 0.5).abs() < 1e-9);
        assert_eq!(stats.plane_one_fractions[3], 0.0);
    }

    #[test]
    fn packed_stats_separate_random_bytes_from_peaked_weights() {
        let random = pseudo_random_bytes(40_000, 7);
        let peaked: Vec<u8> = (0..40_000u32)
            .map(|i| {
                // Quantized weights cluster around the zero point (8); build bytes of two nibbles near 8.
                let a = 8 + (i % 3) as u8;
                let b = 7 + (i % 2) as u8;
                (a << 4) | b
            })
            .collect();
        let random_stats = packed_stats(PackedKind::Bnb4, &random).unwrap();
        let peaked_stats = packed_stats(PackedKind::Bnb4, &peaked).unwrap();
        assert_eq!(random_stats.num_values, 80_000);
        assert!(
            random_stats.value_entropy > 3.95,
            "{}",
            random_stats.value_entropy
        );
        assert!(
            peaked_stats.value_entropy < 2.0,
            "{}",
            peaked_stats.value_entropy
        );
        assert_eq!(random_stats.lane_entropies.len(), 2);
    }

    #[test]
    fn fp8_stats_report_sign_exponent_and_mantissa_fields() {
        let data: Vec<u8> = (0..=255u8).collect();
        let stats = packed_stats(PackedKind::Fp8E4M3, &data).unwrap();
        assert_eq!(stats.num_values, 256);
        // Lanes are the fields [sign, exponent, mantissa].
        assert_eq!(stats.lane_entropies.len(), 3);
        // Every possible byte exactly once: sign is a fair coin, exponent and mantissa are uniform.
        assert!((stats.plane_one_fractions[7] - 0.5).abs() < 1e-9);
        assert!(stats.lane_entropies[1] > 3.9);
    }

    #[test]
    fn packed_stats_reject_what_cannot_be_unpacked() {
        assert!(packed_stats(PackedKind::GptqInt { bits: 3 }, &[0; 8]).is_err());
        // An int32 tensor whose byte length is not a whole number of words.
        assert!(packed_stats(PackedKind::GptqInt { bits: 4 }, &[0; 7]).is_err());
        assert!(packed_stats(PackedKind::GptqInt { bits: 4 }, &[]).is_err());
    }
}
