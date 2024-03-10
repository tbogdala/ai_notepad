#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use anyhow::Context;
use config::Config;
use eframe::egui;
use log::{debug, warn};
use textgen::{TextGenUpdate, TextGeneratorManager};

mod config;
mod textgen;

struct AiNotepadApp {
    config: Config,
    generator: TextGeneratorManager,
}

impl eframe::App for AiNotepadApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::right("right_panel")
            .resizable(false)
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.label("Model ID:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.config.model_id)
                        .hint_text("TheBloke/Llama-2-7B-GGUF"),
                );
                ui.add_space(6.0);

                ui.label("Model File:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.config.model_file)
                        .hint_text("llama-2-7b.Q4_K_M.gguf"),
                );
                ui.add_space(6.0);

                ui.label("Tokenizer ID:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.config.tokenizer_id)
                        .hint_text("TheBloke/Llama-2-7B-fp16"),
                );
                ui.add_space(6.0);

                ui.label("EOS Token:");
                ui.add(
                    egui::TextEdit::singleline(&mut self.config.eos_token_str).hint_text("</s>"),
                );
                ui.add_space(12.0);
                ui.separator();

                ui.label("Generation parameters:");
                ui.add_space(12.0);
                egui::Grid::new("TextgenParams")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Temperature:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.temperature)
                                .speed(0.01)
                                .clamp_range(0.0..=4.0),
                        );
                        ui.end_row();

                        ui.label("Top-P:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.top_p)
                                .speed(0.01)
                                .clamp_range(0.0..=1.0),
                        );
                        ui.end_row();

                        ui.label("Min-P:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.min_p)
                                .speed(0.01)
                                .clamp_range(0.0..=1.0),
                        );
                        ui.end_row();

                        ui.label("Top-K:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.top_k)
                                .speed(1),
                        );
                        ui.end_row();

                        ui.label("Repeat Penalty:");
                        ui.add(
                            egui::DragValue::new(
                                &mut self.config.textgen_parameters.repeat_penalty,
                            )
                            .speed(0.005)
                            .clamp_range(0.0..=2.0),
                        );
                        ui.end_row();

                        ui.label("Repeat Distance:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.repeat_last_n)
                                .speed(1),
                        );
                        ui.end_row();

                        ui.separator();
                        ui.end_row();

                        ui.label("New tokens:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.to_sample)
                                .speed(1.0),
                        );
                        ui.end_row();

                        ui.label("Seed:");
                        ui.add(
                            egui::DragValue::new(&mut self.config.textgen_parameters.seed)
                                .speed(10),
                        );
                        ui.end_row();
                    });

                ui.separator();
                ui.add_space(12.0);
                ui.colored_label(
                    egui::Color32::from_rgb(128, 140, 255),
                    "Press ctrl+Enter to generate. (cmd+Enter on Mac) ",
                );
                ui.add_space(12.0);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // check for input to kick off a text generation
            if ui.input_mut(|i| i.consume_key(egui::Modifiers::COMMAND, egui::Key::Enter)) {
                debug!("Starting text generation!");
                self.generator.generate_text(
                    self.config.model_id.clone(),
                    self.config.model_file.clone(),
                    self.config.tokenizer_id.clone(),
                    self.config.eos_token_str.clone(),
                    self.config.textgen_parameters.clone(),
                    ctx.clone(),
                );
                if let Err(err) = self
                    .config
                    .save()
                    .context("Attempting to save the updated settings to the configuration file")
                {
                    warn!("Failed to save the configuration file: {}", err);
                }
            }

            // check for incoming text generation messages
            if let Some(update_msg) = self.generator.maybe_get_update() {
                // currently we only care about the token updates
                if let TextGenUpdate::Token(tok) = update_msg {
                    self.config
                        .textgen_parameters
                        .user_prompt
                        .push_str(tok.as_str());
                }
            }

            egui::ScrollArea::vertical()
                .auto_shrink(true)
                .min_scrolled_height(64.0)
                .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded)
                .show(ui, |ui| {
                    ui.add_sized(
                        ui.available_size(),
                        egui::TextEdit::multiline(&mut self.config.textgen_parameters.user_prompt)
                            .hint_text("Type a raw prompt to the AI LLM..."),
                    );
                });
        });

        if self.generator.is_busy() {
            egui::TopBottomPanel::bottom("progress").show(ctx, |ui| {
                let (n, d) = self.generator.get_progress();
                let progress = n as f32 / d as f32;

                egui::Grid::new("TextgenParams")
                    .num_columns(2)
                    .show(ui, |ui| {
                        ui.label("Generating:");
                        let progress_bar = egui::ProgressBar::new(progress).show_percentage();
                        ui.add(progress_bar);
                        ui.end_row();
                    });
            });
        }
    }
}

impl AiNotepadApp {
    pub fn new() -> Self {
        AiNotepadApp {
            config: Config::default(),
            generator: TextGeneratorManager::new(),
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    let mut app = AiNotepadApp::new();
    app.config = match Config::load_from_std_location() {
        Ok(c) => c,
        Err(err) => {
            println!("Unable to load the application's config file: {}.", err);
            println!("Using a default configuration file.");
            Config::default()
        }
    };

    // load the application icon file
    let png_data = include_bytes!("../assets/app_icon.png");
    let icon = eframe::icon_data::from_png_bytes(png_data)
        .expect("The embedded application icon should be able to be read from the binary.");

    // setup initial options
    let options = eframe::NativeOptions {
        centered: true,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 480.0])
            .with_icon(icon),
        ..Default::default()
    };

    eframe::run_native(config::APP_TITLE, options, Box::new(|_cc| Box::new(app)))
}
