# AI Notepad v0.2.0

A lightweight application to test interaction with large language models. Currently supports
running GGUF quantized models with hardware acceleration.

![ai_notepad in action](https://github.com/tbogdala/ai_notepad/blob/02674eeaee3304cab95d91b6ccb40a5ca163dd5c/assets/Screenshot.png)

## Features

* Lightweight and simple user interface
* Automatically downloads the necessary files from Huggingface.co
* Easy to play with prompt formatting to see what works and what does not
* Pure [Rust](https://www.rust-lang.org/) implementation using [Candle](https://github.com/huggingface/candle)
  and support Cuda and Metal accelleration
* Serves as an example of a simple Rust app using [egui](https://github.com/emilk/egui) and [Candle](https://github.com/huggingface/candle)


## Known Limitations

* No mirostat sampling or beam sampling.
* No layer control is provided for offloading models to GPU at present.
* It's possible that RTX 3000 series cards or better are the minimum required for Cuda builds - unconfirmed yet.


## Usage

Enter in the full prompt, including whatever text formatting is needed for the particular model and then
hit ctrl+Enter (or cmd+Enter on Macs) to generate text. 


## Notes

* The first time you generate a response with a model it will have to potentially download it from the
  Internet. The download progress will be shared in the terminal window, but the application will otherwise
  not show any progress.
* If you want to see tokenized version of the prompt and generated text, that information is logged, but not
  visible by default. Run the application with the `RUST_LOG=debug` environment variable and value. In the 
  source directory, this could look like `RUST_LOG=debug cargo run --release --features cuda`.


## UI Reference

The 'Model ID' is a Huggingface repository ID such as "TheBloke/Llama-2-7B-GGUF". Similarly, the 'Tokenizer ID'
is another Huggingface repository ID string for the repository that has the `tokenizer.model` file that is
compatible with the model from 'Model ID'. This can be the same repository. The 'Model File' is just the file
name from the Huggingface repository referenced by 'Model ID' such as "llama-2-7b.Q4_K_M.gguf".

The default settings use TheBloke's Llama-2-7B-GGUF Q4_K_M quantized file. Since this model uses the default
Llama-2 tokenizer, it references the f16 version of this model because that's where the tokenizer model lives.
With Llama-2, the default EOS token is `</s>` but other models may change that (e.g. `<|im_end|>`).

So by default, the following values are set:

```yaml
model_id: TheBloke/Llama-2-7B-GGUF
model_file: llama-2-7b.Q4_K_M.gguf
tokenizer_id: TheBloke/Llama-2-7B-fp16
eos_token_str: </s>
```

The "Generation parameters" are all basic text generation parameters. A good reference would be the
[llama.cpp project's documentation of their command-line flags](https://github.com/ggerganov/llama.cpp/tree/master/examples/main#generation-flags), 
which use the same names for things so it directly maps to these parameters.


## Building from source

Both Linux and Windows will require a recent [Cuda installation](https://developer.nvidia.com/cuda-toolkit). The project is also
pure Rust so the [Rust toolchain](https://www.rust-lang.org/learn/get-started) will need to be installed as well.

```bash
git clone https://github.com/tbogdala/ai_notepad.git
cd ai_notepad
cargo build --release --features cuda
```

You can swap out `--features cuda` to `--features metal` for MacOS, or leave it out entirely for a CPU only build. The cuda
binary package for Windows was compiled with both `cuda` and `cudnn` features.

### Windows Build Notes

* The Windows 11 build was tested with Cuda 12.4 and VS 2022. cuDNN was tested with 8.9.6.50 (9.0.0.312 doesn't seem 
  compatible with Candle), using the 'tarball' version so that it could be copied into the Cuda 12.4 folder.
* Make sure to open the `x64 Native Tools Command Prompt` to build the rust project to get the `cuda` feature to build correctly.
