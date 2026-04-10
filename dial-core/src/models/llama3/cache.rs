use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor};

use super::Config;

/// Abstraction over cosine and sine tables, kv-caching and attention masking.
#[derive(Debug, Clone)]
pub struct Cache {
    cos: Tensor,
    sin: Tensor,

    masks: HashMap<usize, Tensor>,
    use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,

    device: Device,
    max_seq_len: usize,
}

impl Cache {
    /// Creates a new cache instance with the provided configuration.
    /// Set `use_kv_cache` to false to disable kv-caching.
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        let max_seq_len = config.max_seq_len;
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;

        log::debug!("cache::n_elem = {n_elem}");

        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rope_theta.powf(i as f32 / n_elem as f32))
            .collect();

        let theta = Tensor::new(theta.as_slice(), device)?;

        log::debug!("cache::theta = {}", &theta);

        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;

        log::debug!("cache::idx_theta = {}", &idx_theta);

        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;

        log::debug!("cache::cos = {}", &cos);
        log::debug!("cache::sin = {}", &sin);

        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            device: device.clone(),
            cos,
            sin,
            max_seq_len,
        })
    }

    /// Return true if kv-caching is enabled.
    pub fn with_kv_cache(&self) -> bool {
        self.use_kv_cache
    }

    /// Return the cached cosine value for the given position and sequence length.
    pub fn cosine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.cos.narrow(0, index_pos, seq_len)
    }

    /// Return the cached sine value for the given position and sequence length.
    pub fn sine(&self, index_pos: usize, seq_len: usize) -> Result<Tensor> {
        self.sin.narrow(0, index_pos, seq_len)
    }

    /// Get the attention mask for the given sequence length.
    pub fn mask(&mut self, seq_len: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&seq_len) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (seq_len, seq_len), &self.device)?;
            self.masks.insert(seq_len, mask.clone());
            Ok(mask)
        }
    }

    /// Process the input k and v by either generating their cache entry or applying a previously cached one.
    /// 这个操作是要更新缓存的
    pub fn process_kv(
        &mut self,
        block_idx: usize,
        index_pos: usize,
        mut k: Tensor,
        mut v: Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if self.use_kv_cache {
            // （新增）当一个“新请求/新对话”的 prefill 开始时，index_pos 会回到 0。
            //
            // 为什么要加：在分布式推理时，master 和 worker 之间是长连接，worker 侧的 KV-cache
            // 如果不在新请求开始时重置，就会把“上一次请求的 k/v”继续 append，导致注意力矩阵的
            // key_len 变大（例如变成 164），但 mask 仍按本次 seq_len（例如 24）生成，从而触发
            // `cannot broadcast [24, 24] to [1, 32, 24, 164]` 这类崩溃。
            //
            // 实现：index_pos==0 时直接覆盖该 block 的 kv 条目，而不是拼接旧缓存。
            if index_pos == 0 {
                self.kvs[block_idx] = Some((k.clone(), v.clone()));
                return Ok((k, v));
            }

            // if this block_idx in cache
            if let Some((cache_k, cache_v)) = &self.kvs[block_idx] {
                // update cache entry
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[2];
                if k_seq_len > self.max_seq_len {
                    k = k
                        .narrow(2, k_seq_len - self.max_seq_len, self.max_seq_len)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[2];
                if v_seq_len > self.max_seq_len {
                    v = v
                        .narrow(2, v_seq_len - self.max_seq_len, self.max_seq_len)?
                        .contiguous()?
                }
            }
            // set entry for this block
            self.kvs[block_idx] = Some((k.clone(), v.clone()))
        }
        Ok((k, v))
    }

    /// Return a copy of this cache with the same state but new kv table.
    pub fn as_new(&self) -> Self {
        let mut copy = self.clone();
        copy.clear();
        copy
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.masks.clear();
        self.kvs = vec![None; self.kvs.len()];
    }

    /// Clone KV tensors for a given layer if present.
    pub fn kv_clone(&self, block_idx: usize) -> Option<(Tensor, Tensor)> {
        self.kvs
            .get(block_idx)
            .and_then(|entry| entry.as_ref().map(|(k, v)| (k.clone(), v.clone())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn test_config(num_layers: usize, max_seq_len: usize) -> Config {
        Config {
            hidden_size: 32,
            intermediate_size: 64,
            vocab_size: 128,
            num_hidden_layers: num_layers,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            bos_token_id: None,
            eos_token_id: None,
            max_seq_len,
        }
    }

    #[test]
    fn process_kv_replaces_on_index_pos_zero() -> Result<()> {
        let device = Device::Cpu;
        let cfg = test_config(2, 64);
        let mut cache = Cache::new(true, DType::F32, &cfg, &device)?;

        // （新增）回归测试：先模拟一次“续写”导致 cache 已存在；
        // 再模拟一次“新请求 prefill（index_pos=0）”，验证不会 append 旧缓存。
        //
        // Shapes: (b, kv_heads, seq, head_dim)
        let k1 = Tensor::zeros((1, 4, 10, 8), DType::F32, &device)?;
        let v1 = Tensor::zeros((1, 4, 10, 8), DType::F32, &device)?;
        let (k, _v) = cache.process_kv(0, 1, k1, v1)?;
        assert_eq!(k.dims()[2], 10);

        // A new prefill should replace instead of append.
        let k2 = Tensor::zeros((1, 4, 7, 8), DType::F32, &device)?;
        let v2 = Tensor::zeros((1, 4, 7, 8), DType::F32, &device)?;
        let (k, _v) = cache.process_kv(0, 0, k2, v2)?;
        assert_eq!(k.dims()[2], 7);

        Ok(())
    }
}
