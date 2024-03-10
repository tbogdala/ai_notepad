use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama as model;
use crossbeam::channel::{Receiver, Sender};
use hf_hub::api::sync::Api;
use log::{debug, error, warn};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextGenerationParams {
    pub to_sample: usize,
    pub temperature: f64,
    pub top_p: f64,
    _top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub user_prompt: String,
}
impl Default for TextGenerationParams {
    fn default() -> Self {
        Self {
            to_sample: 128,
            temperature: 0.2,
            top_p: 0.9,
            _top_k: 40,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 1,
            user_prompt: "".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TextGenUpdate {
    Token(String),
    Finished,
}

pub struct TextGeneratorManager {
    busy_signal: Arc<Mutex<bool>>,
    send: Sender<TextGenUpdate>,
    recv: Receiver<TextGenUpdate>,

    tokens_to_predict: usize,
    tokens_returned: usize,
}
impl TextGeneratorManager {
    pub fn new() -> Self {
        let (send, recv) = crossbeam::channel::unbounded();
        TextGeneratorManager { 
            busy_signal: Arc::new(Mutex::new(false)),
            send, 
            recv ,
            tokens_to_predict: 0,
            tokens_returned: 0,
        }
    }

    // Will return a TextGenUpdate if one is available from the worker thread.
    pub fn maybe_get_update(&mut self) -> Option<TextGenUpdate> {
        if let Ok(update) = self.recv.try_recv() {
            self.tokens_returned += 1;
            Some(update)
        } else {
            None
        }
    }

    // Checks the internal 'busy signal' to see if there's a text
    // generation job already running.
    pub fn is_busy(&self) -> bool {
        let busy = self.busy_signal.lock().unwrap();
        *busy
    }

    // Returns a tuple of (current predicted token count, total count to be predicted).
    pub fn get_progress(&self) -> (usize, usize) {
        (self.tokens_returned, self.tokens_to_predict)
    }

    // Starts a worker thread to handle the text generation and flips
    // the 'busy_signal' mutex so that only one job can run at a time.
    pub fn generate_text(
        &mut self,
        model_id: String,
        model_file: String,
        tokenizer_repo_id: String,
        eos_token_str: String,
        params: TextGenerationParams,
        ctx: eframe::egui::Context,
    ) {
        {
            let mut busy = self.busy_signal.lock().unwrap();
            if *busy == true {
                warn!("Unable to process text generation while already busy.");
                return;
            }

            // set the busy signal
            *busy = true;
        }
        
        let sender_clone = self.send.clone();
        let busy_signal_clone = self.busy_signal.clone();
        
        // make sure to clear the progress tracking
        self.tokens_to_predict = params.to_sample;
        self.tokens_returned = 0;
        
        std::thread::spawn(move || {
            let _ = worker_generate_text(
                model_id.as_str(),
                model_file.as_str(),
                tokenizer_repo_id.as_str(),
                eos_token_str.as_str(),
                &params,
                sender_clone,
                ctx,
            );
            {
                // clear the busy signal
                let mut busy = busy_signal_clone.lock().unwrap();
                *busy = false;
            }
        });
    }
}

// worker function for text generation; does all the actual work.
fn worker_generate_text(
    model_id: &str,
    model_file: &str,
    tokenizer_repo_id: &str,
    eos_token_str: &str,
    params: &TextGenerationParams,
    sender: Sender<TextGenUpdate>,
    ctx: eframe::egui::Context,
) -> Result<String> {
    debug!("generate_text() request started!");

    // get the model filepath from the cache
    let api = Api::new().context("Attempting to create Huggingface API endpoint")?;
    let repo = api.model(model_id.into());
    let model_filepath = repo
        .get(model_file)
        .context("Attempted to get the model weights filepath")?;
    debug!("Using model weights from file: {:?} ...", model_filepath);

    #[cfg(feature = "cuda")]
    let device = Device::new_cuda(0).context("Creating GPU Cuda device for Candle")?;
    #[cfg(feature = "metal")]
    let device = Device::new_metal(0).context("Creating GPU Metal device for Candle")?;
    #[cfg(not(feature = "cuda"))]
    #[cfg(not(feature = "metal"))]
    let device = Device::Cpu;

    // NOTE: Only working with gguf...
    let mut file = std::fs::File::open(&model_filepath)
        .context("Attempting to open the model weights file")?;
    let model =
        gguf_file::Content::read(&mut file).context("Attempting to read the model weights file")?;
    let mut model = model::ModelWeights::from_gguf(model, &mut file, &device)
        .context("Processing model weights")?;
    debug!("Model built successfully.");

    // setup tokenizer
    let tokenizer_repo = api.model(tokenizer_repo_id.into());
    let tokenizer_filepath = tokenizer_repo
        .get("tokenizer.json")
        .context("Attempting to get the tokenizer filepath")?;
    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_filepath)
        .map_err(anyhow::Error::msg)
        .context("Processing model tokenizer")?;
    debug!("Tokenizer deserialized.");

    // NOTE: no safeguarding about overrunning max seq length with sample count!
    let prompt = tokenizer
        .encode(params.user_prompt.clone(), true)
        .map_err(anyhow::Error::msg)
        .context("Tokenizing prompt")?;
    let prompt_tokens = prompt.get_ids();

    // setup an EOS token
    let eos_token = *tokenizer
        .get_vocab(true)
        .get(eos_token_str)
        .context("Attempting to get the EOS token")
        .unwrap();

    // output the prompt tokens to debug log
    let prompt_token_strings = prompt.get_tokens();
    for i in 0..prompt_tokens.len() {
        debug!(
            "Prompt token index {i} is token {} ({})",
            prompt_tokens[i], prompt_token_strings[i]
        );
    }

    let mut all_tokens = vec![];
    let mut logits_processor =
        LogitsProcessor::new(params.seed, Some(params.temperature), Some(params.top_p));
    let start_prompt_processing = std::time::Instant::now();

    // process prompt in one shot
    let input = Tensor::new(prompt_tokens, &device)?.unsqueeze(0)?;
    let logits = model.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;
    let prompt_dt = start_prompt_processing.elapsed();

    let start_post_prompt = std::time::Instant::now();
    let mut sampled = 0;
    let mut prev_full_decode = String::new();
    for index in 0..params.to_sample {
        let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, prompt_tokens.len() + index)?;
        let logits = logits.squeeze(0)?;
        let logits = if params.repeat_penalty == 1. {
            logits
        } else {
            let start_at = all_tokens.len().saturating_sub(params.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                params.repeat_penalty,
                &all_tokens[start_at..],
            )?
        };
        next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);
        sampled += 1;

        // see if we can decode the token to send it out in an update
        if let Ok(update_str) = tokenizer.decode(&all_tokens, false) {
            // Gonna do this in the most naive, braindead way possible: decode the whole incoming
            // buffer and then send new tail off. Doing the decode one at a time skips all the spacing...
            let (_, new_tail) = update_str.split_at(prev_full_decode.len());
            debug!("DEBUG: sending update: {}", new_tail);
            if let Err(err) = sender.send(TextGenUpdate::Token(new_tail.to_string())) {
                error!(
                    "Error while sending text generation update message along channel: {}",
                    err
                );
            }
            ctx.request_repaint();
            prev_full_decode = update_str;
        }

        if next_token == eos_token {
            break;
        };
    }
    let dt = start_post_prompt.elapsed();
    debug!(
        "\n\n{:4} prompt tokens processed: {:.2} token/s",
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prompt_dt.as_secs_f64(),
    );
    debug!(
        "{sampled:4} tokens generated: {:.2} token/s",
        sampled as f64 / dt.as_secs_f64(),
    );

    // writing out all the generated tokens to debug log
    for i in 0..all_tokens.len() {
        let t_str = tokenizer.decode(&all_tokens[i..i + 1], false).unwrap();
        debug!(
            "generated token index {i} is token {} ({})",
            all_tokens[i], t_str
        );
    }

    let whole_string = tokenizer
        .decode(&all_tokens.as_slice(), false)
        .map_err(anyhow::Error::msg)
        .context("Decoding the predicted text");
    if let Ok(decoded) = &whole_string {
        debug!("Whole predicted text is:");
        debug!("{}", decoded);
    }

    if let Err(err) = sender.send(TextGenUpdate::Finished) {
        error!(
            "Error while sending finish message to text generator channel: {}",
            err
        );
    }
    ctx.request_repaint();

    whole_string
}
