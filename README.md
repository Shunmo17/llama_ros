llama_ros
===

This is a ROS 1 wrapper for llama.cpp. Original is [llama_ros](https://github.com/mgonzs13/llama_ros).
This package support only llama2. If you want to use other LLMs, please add launch files for them.

## Requirement

- ROS1 Noetic

## Usage

### Launch llama2 server.

**Run on GPU**

```bash
roslaunch llama_bringup llama2.launch
```

**run on CPU**

```bash
roslaunch llama_bringup llama2.launch n_gpu_layers:=0
```

### Send prompt

```bash
roslaunch llama_ros llama_client_node.launch prompt:=<YOUR PROMPT>
```

## Installation

Firstly, you have to an environment for llama2.
Then, please follow the instalation procudure.

### Setup workspace

Please modify the workspace path to adapt ypur environment.

```bash
mkdir -p workspace/src
cd workspace/src
git clone --recursive git@github.com:Shunmo17/llama_ros.git
catkin b
```

### Download models

Llama2 models finetuned for chat are available here: [7B](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main), [13B](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main), [70B](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGML/tree/main)

After downloding a model, please set the correct model path in `llama2.launch`.

## Parameters

- `use_default_sampling_config`

  If it is true, we ignore the samping config in the requested goal. This parameter is added not to use uninitialized parameters because we cannot set default parameters of action messages in ROS1.

## Maintainer

[Shunmo17](https://github.com/Shunmo17)
