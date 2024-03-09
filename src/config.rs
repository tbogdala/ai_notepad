use std::path::PathBuf;

use anyhow::{Context, Result};
use dirs::config_dir;
use log::debug;
use serde::{Deserialize, Serialize};

use crate::textgen::TextGenerationParams;

pub const APP_TITLE: &str = "AI Notepad";
pub const APP_CONFIG_FOLDER: &str = "ai_notepad";
pub const APP_CONFIG_FILE: &str = "config.yaml";

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub model_id: String,
    pub model_file: String,
    pub tokenizer_id: String,
    pub eos_token_str: String,
    pub textgen_parameters: TextGenerationParams,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            model_id: "TheBloke/Llama-2-7B-GGUF".into(),
            model_file: "llama-2-7b.Q4_K_M.gguf".into(),
            tokenizer_id: "TheBloke/Llama-2-7B-fp16".into(),
            eos_token_str: "</s>".into(),
            textgen_parameters: Default::default(),
        }
    }
}

impl Config {
    fn get_std_config_filepath() -> Result<PathBuf> {
        let mut config_path = config_dir().context("Getting the configuration directory")?;
        config_path.push(APP_CONFIG_FOLDER);
        config_path.push(APP_CONFIG_FILE);
        Ok(config_path)
    }

    pub fn load_from_std_location() -> Result<Self> {
        let config_filepath = Self::get_std_config_filepath()?;
        debug!(
            "Attempting to load configuration file from: {:?}.",
            config_filepath
        );
        let plain_string = std::fs::read_to_string(&config_filepath).context(format!(
            "Reading configuration file at {:?}",
            config_filepath
        ))?;
        let cfg = serde_yaml::from_str::<Config>(plain_string.as_str()).context(format!(
            "Deserializing YAML configuration file at {:?}",
            config_filepath
        ))?;

        Ok(cfg)
    }

    pub fn save(&self) -> Result<()> {
        let config_filepath = Self::get_std_config_filepath()?;
        debug!(
            "Attempting to save configuration file to: {:?}.",
            config_filepath
        );

        // Ensure parent directories exist before writing the file
        std::fs::create_dir_all(
            config_filepath
                .parent()
                .context("Getting parent folder for configuration file")?,
        )
        .context("Creating the folder structure for the configuration file in the file system")?;

        let yaml_string =
            serde_yaml::to_string(self).context("Serializing the configuration data")?;
        std::fs::write(config_filepath, yaml_string)
            .context("Writing serialized configuration to the file system")?;
        Ok(())
    }
}
