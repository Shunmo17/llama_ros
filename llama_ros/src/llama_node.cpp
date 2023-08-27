// MIT License

// Copyright (c) 2023  Miguel Ángel González Santamarta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include <memory>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>
#include <thread>

#include "llama.h"
#include "llama_msgs/TokenProb.h"
#include "llama_msgs/TokenProbArray.h"
#include "llama_ros/llama_node.hpp"

using namespace llama_ros;

LlamaNode::LlamaNode()
    : pnh_("~") {
  std::string model;
  std::string lora_adapter;
  std::string lora_base;
  bool numa;
  std::string prefix;
  std::string suffix;
  std::string stop;

  std::vector<double> tensor_split;

  std::string prompt;
  std::string file_path;

  pnh_.param("use_default_sampling_config", use_default_sampling_config_, false);

  auto context_params = llama_context_default_params();
  auto eval_params = llama_eval_default_params();

  int seed_tmp;
  pnh_.param("seed", seed_tmp, -1);
  context_params.seed = static_cast<uint32_t>(seed_tmp);

  pnh_.param("n_ctx", context_params.n_ctx, 512);
  pnh_.param("n_batch", context_params.n_batch, 512);
  pnh_.param("n_gqa", context_params.n_gqa, 1);
  pnh_.param("rms_norm_eps", context_params.rms_norm_eps, 5e-6f);

  pnh_.param("n_gpu_layers", context_params.n_gpu_layers, 0);
  pnh_.param("main_gpu", context_params.main_gpu, 0);
  pnh_.param("tensor_split", tensor_split, {0.0});

  pnh_.param("rope_freq_scale", context_params.rope_freq_scale, 1.0f);
  pnh_.param("rope_freq_base", context_params.rope_freq_base, 10000.0f);

  pnh_.param("low_vram", context_params.low_vram, false);
  pnh_.param("mul_mat_q", context_params.mul_mat_q, false);
  pnh_.param("f16_kv", context_params.f16_kv, true);
  pnh_.param("logits_all", context_params.logits_all, false);
  pnh_.param("vocab_only", context_params.vocab_only, false);
  pnh_.param("use_mmap", context_params.use_mmap, true);
  pnh_.param("use_mlock", context_params.use_mlock, false);
  pnh_.param("embedding", context_params.embedding, true);

  pnh_.param("n_threads", eval_params.n_threads, 1);
  pnh_.param("n_predict", eval_params.n_predict, 128);
  pnh_.param("n_keep", eval_params.n_keep, -1);
  pnh_.param("n_batch", eval_params.n_batch);

  pnh_.param("model", model, static_cast<std::string>(""));
  pnh_.param("lora_adapter", lora_adapter, static_cast<std::string>(""));
  pnh_.param("lora_base", lora_base, static_cast<std::string>(""));
  pnh_.param("numa", numa, false);

  pnh_.param("prefix", prefix, static_cast<std::string>(""));
  pnh_.param("suffix", suffix, static_cast<std::string>(""));
  pnh_.param("stop", stop, static_cast<std::string>(""));

  pnh_.param("prompt", prompt, static_cast<std::string>(""));
  pnh_.param("file", file_path, static_cast<std::string>(""));

  // parse tensor_split
  context_params.tensor_split =
      reinterpret_cast<const float *>(tensor_split.data());

  // check threads number
  if (eval_params.n_threads < 0) {
    eval_params.n_threads = std::thread::hardware_concurrency();
  }

  // load llama
  this->llama =
      std::make_shared<Llama>(context_params, eval_params, model, lora_adapter,
                              lora_base, numa, prefix, suffix, stop);

  // initial prompt
  if (!file_path.empty()) {
    std::ifstream file(file_path.c_str());
    if (!file) {
      ROS_ERROR_STREAM("Failed to open file " << file_path);
    }
    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(), back_inserter(prompt));
  }
  this->llama->generate_response(prompt, false, llama_sampling_default_params(),
                                 nullptr);

  // services
  this->tokenize_service_ = nh_.advertiseService(
      "tokenize", &LlamaNode::tokenize_service_callback, this);
  this->generate_embeddings_service_ = nh_.advertiseService(
      "generate_embeddings", &LlamaNode::generate_embeddings_service_callback,
      this);

  // start reponse action server
  generate_response_action_server_ = std::make_unique<
      actionlib::SimpleActionServer<llama_msgs::GenerateResponseAction>>(
      nh_, "generate_response", boost::bind(&LlamaNode::execute, this, _1),
      false);
  generate_response_action_server_->start();

  ROS_INFO_STREAM("Llama Node started");
}

bool LlamaNode::tokenize_service_callback(
    llama_msgs::Tokenize::Request &request,
    llama_msgs::Tokenize::Response &response) {

  response.tokens = this->llama->tokenize(request.prompt, false);
  return true;
}

bool LlamaNode::generate_embeddings_service_callback(
    llama_msgs::GenerateEmbeddings::Request &request,
    llama_msgs::GenerateEmbeddings::Response &response) {

  if (this->llama->is_embedding()) {
    response.embeddings = this->llama->generate_embeddings(request.prompt);
  }
  return true;
}

void LlamaNode::execute(
    const llama_msgs::GenerateResponseGoalConstPtr &goal) {

  // get goal data
  std::string prompt = goal->prompt;
  bool reset = goal->reset;
  auto sampling_config = goal->sampling_config;

  auto result = std::make_shared<llama_msgs::GenerateResponseResult>();

  ROS_INFO_STREAM("Prompt received: " << prompt);

  // reset llama
  if (reset) {
    this->llama->reset();
  }

  if (use_default_sampling_config_) {
    sampling_config.ignore_eos = false;
    sampling_config.temp = 0.80;
    sampling_config.top_k = 40;
    sampling_config.top_p = 0.95;
    sampling_config.tfs_z = 1.00;
    sampling_config.typical_p = 1.00;
    sampling_config.repeat_last_n = 64;
    sampling_config.repeat_penalty = 1.10;
    sampling_config.presence_penalty = 0.00;
    sampling_config.frequency_penalty = 0.00;
    sampling_config.mirostat = 0;
    sampling_config.mirostat_eta = 0.10;
    sampling_config.mirostat_tau = 5.0;
    sampling_config.penalize_nl = true;
    sampling_config.n_probs = 1;
    sampling_config.grammar = "";
  }

  // prepare sampling params
  auto sampling_params = llama_sampling_default_params();
  sampling_params.ignore_eos = sampling_config.ignore_eos;
  sampling_params.temp = sampling_config.temp;
  sampling_params.top_k = sampling_config.top_k;
  sampling_params.top_p = sampling_config.top_p;
  sampling_params.tfs_z = sampling_config.tfs_z;
  sampling_params.typical_p = sampling_config.typical_p;
  sampling_params.repeat_last_n = sampling_config.repeat_last_n;
  sampling_params.repeat_penalty = sampling_config.repeat_penalty;
  sampling_params.presence_penalty = sampling_config.presence_penalty;
  sampling_params.frequency_penalty = sampling_config.frequency_penalty;
  sampling_params.mirostat = sampling_config.mirostat;
  sampling_params.mirostat_eta = sampling_config.mirostat_eta;
  sampling_params.mirostat_tau = sampling_config.mirostat_tau;
  sampling_params.penalize_nl = sampling_config.penalize_nl;
  sampling_params.n_probs = sampling_config.n_probs;
  sampling_params.grammar = sampling_config.grammar;

  // check repeat_last_n
  sampling_params.repeat_last_n = sampling_params.repeat_last_n < 0
                                      ? this->llama->get_n_ctx()
                                      : sampling_params.repeat_last_n;

  // check top_k
  sampling_params.top_k = sampling_params.top_k <= 0
                              ? this->llama->get_n_vocab()
                              : sampling_params.top_k;

  // add logit bias
  for (auto logit_bias : sampling_config.logit_bias.data) {
    sampling_params.logit_bias[logit_bias.token] = logit_bias.bias;
  }

  // add llama_token_eos
  if (sampling_params.ignore_eos) {
    sampling_params.logit_bias[llama_token_eos()] = -INFINITY;
  }

  // call llama
  auto completion_results = this->llama->generate_response(
      prompt, true, sampling_params,
      std::bind(&LlamaNode::send_text, this, std::placeholders::_1));

  for (auto completion : completion_results) {
    result->response.text.append(this->llama->detokenize({completion.token}));
    result->response.tokens.push_back(completion.token);

    llama_msgs::TokenProbArray probs_msg;
    for (auto prob : completion.probs) {
      llama_msgs::TokenProb aux;
      aux.token = prob.token;
      aux.probability = prob.probability;
      aux.token_text = this->llama->detokenize({prob.token});
      probs_msg.data.push_back(aux);
    }
    result->response.probs.push_back(probs_msg);
  }

  if (ros::ok()) {
    if (generate_response_action_server_->isPreemptRequested()) {
      generate_response_action_server_->setPreempted(*result);
    } else {
      generate_response_action_server_->setSucceeded(*result);
    }
  }
}

void LlamaNode::send_text(const completion_output &completion) {

  auto feedback = std::make_shared<llama_msgs::GenerateResponseFeedback>();

  feedback->partial_response.text =
      this->llama->detokenize({completion.token});
  feedback->partial_response.token = completion.token;
  feedback->partial_response.probs.chosen_token = completion.token;

  for (auto prob : completion.probs) {
    llama_msgs::TokenProb aux;
    aux.token = prob.token;
    aux.probability = prob.probability;
    aux.token_text = this->llama->detokenize({prob.token});
    feedback->partial_response.probs.data.push_back(aux);
  }

  generate_response_action_server_->publishFeedback(*feedback);
}

int main(int argc, char *argv[]) {
  ros::init(argc, argv, "llama_node");
  LlamaNode llama_node;
  ros::spin();
  return 0;
}
