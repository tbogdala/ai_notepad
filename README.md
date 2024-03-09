# AI Notepad

A lightweight application to test interaction with large language models. Currently supports
running GGUF quantized models with hardware acceleration.

![ai_notepad in action](https://github.com/tbogdala/ai_notepad/blob/9e398200df923c50adbce5733ec0b8d93db73c85/assets/Screenshot.png)

## Features

* Lightweight and simple user interface
* Automatically downloads the necessary files from Huggingface.co
* Easy to play with prompt formatting to see what works and what does not
* Pure [Rust](https://www.rust-lang.org/) implementation using [Candle](https://github.com/huggingface/candle)
  and support Cuda and Metal accelleration
* Serves as an example of a simple Rust app using [egui](https://github.com/emilk/egui) and [Candle](https://github.com/huggingface/candle)


## Known Limitations

* The sampler implementation is cribbed from Candle's examples and is *very basic*; this will be replaced soon.
* No layer control is provided for offloading models to GPU at present.


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

You can swap out `--features cuda` to `--features metal` for MacOS, or leave it out entirely for a CPU only build.

### Windows Build Notes

* The Windows 11 build was tested with Cuda 12.4 and VS 2022. cuDNN was tested (unsuccessfully) with 9.0.0, using the 'tarball' version so that
  it could be copied into the Cuda 12.4 folder.
* Make sure to open the `x64 Native Tools Command Prompt` to build the rust project to get the `cuda` feature to build correctly.

### Windows WSL Build Notes

Candle still doesn't build with CUDA accelleration by default in Windows, so everything has to be installed 
via WSL.

1. Install WSL 2 using the [install notes from Microsoft](https://learn.microsoft.com/en-us/windows/wsl/install-manual).
   Because the NVidia package in a later step is labeled for Ubuntu, I went ahead and installed the 'Ubuntu' app from the Microsoft Store.

2. Run the newly installed 'Ubuntu' app as Administrator. Setup your user, login and do the initial `sudo apt update;sudo apt upgrade` pass.

3. Install some handy dependencies: `sudo apt install git vim wget build-essential libssl-dev pkg-config`

4. Install [Rust](https://www.rust-lang.org/learn/get-started) using their script: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   and don't forget to `source ~/.bashrc` when done to refresh your shell.

5. At this point you should be able to build the Candle examples without CUDA accelleration in WSL, but to get CUDA accelleration working,
   follow [these steps from NVidia] (https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
   to install that specific type of package. Make sure to add the `bin` folder to your `$PATH` by running something like
   `export PATH=$PATH:/usr/local/cuda-12.4/bin` or modifying your `~/.bashrc`.

6. Now the Candle examples should build with CUDA accelleration. But we want more power! For that we need to get the cudnn library setup:

```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
tar vxf cudnn-linux-x86_64-9.0.0.312_cuda12-archive.tar.xz
sudo mv -v cudnn-linux-x86_64-9.0.0.312_cuda12-archive/include/* /usr/local/cuda-12.4/include 
sudo mv -v cudnn-linux-x86_64-9.0.0.312_cuda12-archive/lib/* /usr/local/cuda-12.4/lib64
sudo chmod a+r /usr/local/cuda-12.4/lib64/libcudnn* 
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH"
```


