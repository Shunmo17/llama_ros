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

#include <cassert>
#include <cmath>
#include <thread>

#include "llama_ros/llama.hpp"

using namespace llama_ros;

struct llama_sampling_params llama_sampling_default_params() {
  struct llama_sampling_params result = {
      /*ignore_eos              =*/false,
      /*logit_bias              =*/{},
      /*.temp                   =*/0.80f,
      /*.top_k                  =*/40,
      /*.top_p                  =*/0.95f,
      /*.tfs_z                  =*/1.00f,
      /*.typical_p              =*/1.00f,
      /*.repeat_penalty         =*/1.10f,
      /*.repeat_last_n          =*/64,
      /*.presence_penalty       =*/0.00f,
      /*.frequency_penalty      =*/0.00f,
      /*.mirostat               =*/0,
      /*.mirostat_tau           =*/5.0f,
      /*.mirostat_eta           =*/0.10f,
      /*.penalize_nl            =*/true,
      /*.n_probs                =*/0,
      /*grammar                 =*/"",
  };
  return result;
}

struct llama_eval_params llama_eval_default_params() {
  struct llama_eval_params result = {
      /*.n_threads              =*/4,
      /*.n_predict              =*/128,
      /*.n_batch                =*/8,
      /*.n_keep                 =*/-1,
  };
  return result;
}

Llama::Llama(llama_context_params context_params,
             const llama_eval_params &eval_params, const std::string &model,
             const std::string &lora_adapter, const std::string &lora_base,
             const bool &numa, const std::string &prefix,
             const std::string &suffix, const std::string &stop)
    : eval_params(eval_params), stop(stop) {

  this->embedding = context_params.embedding;

#ifdef GGML_USE_CUBLAS
  context_params.low_vram = true;
#endif

  if (context_params.rope_freq_base != 10000.0) {
    fprintf(stderr,
            "warning: changing RoPE frequency base to %g (default 10000.0)\n",
            context_params.rope_freq_base);
  }

  if (context_params.rope_freq_scale != 1.0) {
    fprintf(stderr, "warning: scaling RoPE frequency by %g (default 1.0)\n",
            context_params.rope_freq_scale);
  }

  if (context_params.n_ctx > 2048) {
    fprintf(stderr,
            "warning: base model only supports context sizes no greater "
            "than 2048 tokens (%d specified)\n",
            context_params.n_ctx);
  } else if (context_params.n_ctx < 8) {
    fprintf(stderr,
            "warning: minimum context size is 8, using minimum size.\n");
    context_params.n_ctx = 8;
  }

  // load the model
  llama_backend_init(numa);

  this->model = llama_load_model_from_file(model.c_str(), context_params);
  if (this->model == NULL) {
    fprintf(stderr, "Failed to load model '%s'\n", model.c_str());
  }

  this->ctx = llama_new_context_with_model(this->model, context_params);
  if (this->ctx == NULL) {
    fprintf(stderr, "Failed to create context with model '%s'\n",
            model.c_str());
  }

  if (!lora_adapter.empty()) {
    if (llama_model_apply_lora_from_file(this->model, lora_adapter.c_str(),
                                         lora_base.empty() ? NULL
                                                           : lora_base.c_str(),
                                         this->eval_params.n_threads)) {
      fprintf(stderr, "Failed to apply lora adapter\n");
    }
  }

  // show system information
  fprintf(stderr, "System_info: n_threads = %d / %d | %s\n",
          this->eval_params.n_threads, std::thread::hardware_concurrency(),
          llama_print_system_info());

  // prefix & suffix
  this->inp_pfx = this->tokenize(prefix, true);
  this->inp_sfx = this->tokenize(suffix, false);
  this->inp_stop = this->tokenize(stop, false);

  // number of tokens to keep when resetting context
  if (this->eval_params.n_keep == -1) {
    this->eval_params.n_keep = (int)this->prompt_tokens.size();
  }

  // TODO: replace with ring-buffer
  this->last_n_tokens = std::vector<llama_token>(this->get_n_ctx());
  std::fill(this->last_n_tokens.begin(), this->last_n_tokens.end(), 0);

  this->is_antiprompt = false;
  this->canceled = false;
  this->n_past = 0;
  this->n_remain = this->eval_params.n_predict;
  this->n_consumed = 0;

  // show info
  fprintf(stderr,
          "Generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n",
          this->get_n_ctx(), this->eval_params.n_batch,
          this->eval_params.n_predict, this->eval_params.n_keep);

  // do one empty run to warm up the model
  {
    const std::vector<llama_token> tmp = {
        llama_token_bos(),
    };
    llama_eval(this->ctx, tmp.data(), tmp.size(), 0,
               this->eval_params.n_threads);
    llama_reset_timings(this->ctx);
  }
}

Llama::~Llama() {
  llama_free(this->ctx);
  llama_free_model(this->model);
  llama_backend_free();
}

std::vector<llama_token> Llama::tokenize(const std::string &text,
                                         bool add_bos) {
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<llama_token> res(text.size() + (int)add_bos);
  int n =
      llama_tokenize(this->ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

std::string Llama::detokenize(const std::vector<llama_token> &tokens) {
  std::string output = "";
  for (llama_token token : tokens) {
    output += llama_token_to_str(this->ctx, token);
  }
  return output;
}

void Llama::reset() {
  this->last_n_tokens = std::vector<llama_token>(this->get_n_ctx());
  std::fill(this->last_n_tokens.begin(), this->last_n_tokens.end(), 0);

  this->is_antiprompt = false;
  this->canceled = false;
  this->n_past = 0;
  this->n_remain = this->eval_params.n_predict;
  this->n_consumed = 0;

  this->prompt_tokens.clear();
  this->batch_tokens.clear();
}

void Llama::cancel() { this->canceled = true; }

std::vector<float> Llama::generate_embeddings(const std::string &input_prompt) {

  if (!this->is_embedding()) {
    fprintf(stderr,
            "Llama must be created with embedding=true to create embeddings\n");
    return {};
  }

  std::string prompt(input_prompt);
  prompt.insert(0, 1, ' ');
  auto tokens = this->tokenize(prompt, true);
  int n_past = 0;

  for (int i = 0; i < (int)tokens.size(); i += this->eval_params.n_batch) {

    int n_eval = (int)tokens.size() - i;
    if (n_eval > this->eval_params.n_batch) {
      n_eval = this->eval_params.n_batch;
    }

    if (llama_eval(this->ctx, &tokens[i], n_eval, n_past,
                   this->eval_params.n_threads)) {
      fprintf(stderr, "Failed to eval\n");
    }
    n_past += n_eval;
  }

  const int n_embd = llama_n_embd(this->ctx);
  const auto embeddings = llama_get_embeddings(this->ctx);
  std::vector<float> embeddings_list;

  for (int i = 0; i < n_embd; i++) {
    embeddings_list.push_back(embeddings[i]);
  }

  return embeddings_list;
}

std::vector<completion_output>
Llama::generate_response(const std::string &input_prompt, bool add_pfx_sfx,
                         const llama_sampling_params &sampling_params,
                         GenerateResponseCallback callback) {

  this->canceled = false;
  bool input_noecho = true;

  bool stopping = false;
  completion_output completion_result;
  std::vector<completion_output> response;
  std::vector<completion_output> completion_result_list;

  std::string prompt(input_prompt);
  std::vector<llama_token> line_inp;

  if (prompt.size() <= 0) {
    return {};
  }

  if (!this->prompt_tokens.size()) {
    prompt.insert(0, 1, ' ');
  }

  if (!this->prompt_tokens.size() && !add_pfx_sfx) {
    line_inp = this->tokenize(prompt, true);
  } else {
    line_inp = this->tokenize(prompt, false);
  }

  int prompt_size = this->prompt_tokens.size() + line_inp.size();
  if (add_pfx_sfx) {
    prompt_size += this->inp_pfx.size() + this->inp_sfx.size();
  }

  if (prompt_size > this->get_n_ctx() - 4) {
    fprintf(stderr, "Prompt is too long (%d tokens, max %d)\n", prompt_size,
            this->get_n_ctx() - 4);
  }

  // insert prefix
  if (add_pfx_sfx && this->inp_pfx.size() && !this->is_antiprompt) {
    this->prompt_tokens.insert(this->prompt_tokens.end(), this->inp_pfx.begin(),
                               this->inp_pfx.end());
  }

  this->prompt_tokens.insert(this->prompt_tokens.end(), line_inp.begin(),
                             line_inp.end());

  // insert suffix
  if (add_pfx_sfx && this->inp_sfx.size()) {
    this->prompt_tokens.insert(this->prompt_tokens.end(), this->inp_sfx.begin(),
                               this->inp_sfx.end());
  }

  this->n_remain -= line_inp.size();

  // show sampling info
  fprintf(stderr,
          "Sampling: temp = %f, "
          "top_k = %d, "
          "top_p = %f, "
          "repeat_last_n = %i, "
          "repeat_penalty = %f\n",
          sampling_params.temp, sampling_params.top_k, sampling_params.top_p,
          sampling_params.repeat_last_n, sampling_params.repeat_penalty);

  // load grammar
  this->grammar = this->load_grammar(sampling_params.grammar);

  if (this->grammar != NULL) {
    auto it = sampling_params.logit_bias.find(llama_token_eos());

    if (it != sampling_params.logit_bias.end() && it->second == -INFINITY) {
      fprintf(stderr, "warning: EOS token is disabled, which will cause most "
                      "grammars to fail\n");
    }
  }

  fprintf(stderr, "Starting Response Generation\n");

  // generation loop
  while (this->n_remain != 0) {

    this->eval();

    if ((int)this->prompt_tokens.size() <= this->n_consumed) {

      // check if stop appears at the end of the output
      std::string last_output = this->detokenize(this->last_n_tokens);
      this->is_antiprompt = false;

      // when not currently processing queued
      // inputs check if we should end
      if (this->stop.size() &&
          last_output.find(this->stop.c_str(),
                           last_output.length() - this->stop.length(),
                           this->stop.length()) != std::string::npos) {
        this->is_antiprompt = true;
        break;
      }

      // sample next token
      completion_result = this->sample(sampling_params);
      completion_result_list.push_back(completion_result);

      this->batch_tokens.push_back(completion_result.token);
      this->last_n_tokens.erase(this->last_n_tokens.begin());
      this->last_n_tokens.push_back(completion_result.token);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --this->n_remain;
    }

    if (this->batch_tokens.back() == llama_token_eos()) {
      break;
    }

    if (this->canceled) {
      fprintf(stderr, "Canceling llama.cpp\n");
      break;
    }

    // check if new tokens contains the stop sequence
    if (completion_result_list.size() <= this->inp_stop.size()) {

      stopping = true;

      for (size_t i = 0; i < (size_t)completion_result_list.size(); i++) {
        if (completion_result_list.at(i).token != this->inp_stop.at(i)) {
          stopping = false;
          break;
        }
      }

      if (stopping && completion_result_list.size() == this->inp_stop.size()) {
        break;
      }

    } else {
      stopping = false;
    }

    // send text
    if (!input_noecho) {
      if (!stopping) {
        for (auto completion_ele : completion_result_list) {
          if (callback != nullptr) {
            callback(completion_ele);
          }
          response.push_back(completion_ele);
        }
        completion_result_list.clear();
      }
    }

    // respect the maximum number of tokens
    if (this->n_remain <= 0 && this->eval_params.n_predict != -1) {
      this->n_remain = this->eval_params.n_predict;
      break;
    }
  }

  fprintf(stderr, "Finish Response Generation\n");

  if (this->grammar != NULL) {
    llama_grammar_free(this->grammar);
    this->grammar = NULL;
  }

  return response;
}

void Llama::eval() {

  while (((int)this->prompt_tokens.size() > this->n_consumed) &&
         ((int)this->batch_tokens.size() < this->eval_params.n_batch)) {

    this->batch_tokens.push_back(this->prompt_tokens[this->n_consumed]);
    this->last_n_tokens.erase(this->last_n_tokens.begin());
    this->last_n_tokens.push_back(this->prompt_tokens[this->n_consumed]);
    ++this->n_consumed;
  }

  // predict
  if (this->batch_tokens.size() > 0) {

    // infinite text generation via context swapping
    // if we run out of context:
    // - take the n_keep first tokens from the original prompt (via n_past)
    // - take half of the last (n_ctx - n_keep) tokens and recompute the
    // logits in a batch
    if (this->n_past + (int)this->batch_tokens.size() > this->get_n_ctx()) {

      const int n_left = this->n_past - this->eval_params.n_keep;
      this->n_past = this->eval_params.n_keep;

      // insert n_left/2 tokens at the start of batch_tokens
      // from last_n_tokens
      this->batch_tokens.insert(
          this->batch_tokens.begin(),
          this->last_n_tokens.begin() + this->get_n_ctx() - n_left / 2 -
              this->batch_tokens.size(),
          this->last_n_tokens.end() - this->batch_tokens.size());
    }

    // evaluate tokens in batches
    // batch_tokens is typically prepared beforehand to fit within a batch
    // but not always
    for (int i = 0; i < (int)this->batch_tokens.size();
         i += this->eval_params.n_batch) {

      int n_eval = (int)this->batch_tokens.size() - i;
      if (n_eval > this->eval_params.n_batch) {
        n_eval = this->eval_params.n_batch;
      }

      spinner.spin("EVALUATING " + std::to_string(n_eval) + " TOKENS");

      if (llama_eval(this->ctx, &this->batch_tokens[i], n_eval, this->n_past,
                     this->eval_params.n_threads)) {
        fprintf(stderr, "Failed to eval\n");
      }
      this->n_past += n_eval;
    }

    this->batch_tokens.clear();
  }
}

completion_output Llama::sample(llama_sampling_params sampling_params) {

  // init token
  llama_token id = 0;
  auto logits = llama_get_logits(this->ctx);
  auto n_vocab = llama_n_vocab(this->ctx);

  // apply logit_bias
  for (auto it = sampling_params.logit_bias.begin();
       it != sampling_params.logit_bias.end(); it++) {
    logits[it->first] += it->second;
  }

  // candidates
  std::vector<llama_token_data> candidates;
  candidates.reserve(n_vocab);
  for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
    candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
  }

  llama_token_data_array candidates_p = {candidates.data(), candidates.size(),
                                         false};

  // Apply penalties
  float nl_logit = logits[llama_token_nl()];
  auto last_n_repeat = std::min(
      std::min((int)this->last_n_tokens.size(), sampling_params.repeat_last_n),
      this->get_n_ctx());

  llama_sample_repetition_penalty(
      this->ctx, &candidates_p,
      this->last_n_tokens.data() + this->last_n_tokens.size() - last_n_repeat,
      last_n_repeat, sampling_params.repeat_penalty);
  llama_sample_frequency_and_presence_penalties(
      this->ctx, &candidates_p,
      this->last_n_tokens.data() + this->last_n_tokens.size() - last_n_repeat,
      last_n_repeat, sampling_params.frequency_penalty,
      sampling_params.presence_penalty);

  if (!sampling_params.penalize_nl) {
    logits[llama_token_nl()] = nl_logit;
  }

  if (this->grammar != NULL) {
    llama_sample_grammar(this->ctx, &candidates_p, this->grammar);
  }

  if (sampling_params.temp <= 0) {

    // Greedy sampling
    id = llama_sample_token_greedy(this->ctx, &candidates_p);

    if (sampling_params.n_probs > 0) {
      llama_sample_softmax(this->ctx, &candidates_p);
    }

  } else {
    if (sampling_params.mirostat == 1) {
      static float mirostat_mu = 2.0f * sampling_params.mirostat_tau;
      const int mirostat_m = 100;
      llama_sample_temperature(this->ctx, &candidates_p, sampling_params.temp);
      id = llama_sample_token_mirostat(
          this->ctx, &candidates_p, sampling_params.mirostat_tau,
          sampling_params.mirostat_eta, mirostat_m, &mirostat_mu);

    } else if (sampling_params.mirostat == 2) {
      static float mirostat_mu = 2.0f * sampling_params.mirostat_tau;
      llama_sample_temperature(this->ctx, &candidates_p, sampling_params.temp);
      id = llama_sample_token_mirostat_v2(
          this->ctx, &candidates_p, sampling_params.mirostat_tau,
          sampling_params.mirostat_eta, &mirostat_mu);

    } else {

      // Temperature sampling
      size_t min_keep = std::max(1, sampling_params.n_probs);

      llama_sample_top_k(this->ctx, &candidates_p, sampling_params.top_k,
                         min_keep);
      llama_sample_tail_free(this->ctx, &candidates_p, sampling_params.tfs_z,
                             min_keep);
      llama_sample_typical(this->ctx, &candidates_p, sampling_params.typical_p,
                           min_keep);
      llama_sample_top_p(this->ctx, &candidates_p, sampling_params.top_p,
                         min_keep);
      llama_sample_temperature(this->ctx, &candidates_p, sampling_params.temp);
      id = llama_sample_token(this->ctx, &candidates_p);
    }
  }

  if (this->grammar != NULL) {
    llama_grammar_accept_token(this->ctx, this->grammar, id);
  }

  // create output
  completion_output result;
  result.token = id;

  for (size_t i = 0;
       i < std::min(candidates_p.size, (size_t)sampling_params.n_probs); ++i) {
    result.probs.push_back({candidates_p.data[i].id, candidates_p.data[i].p});
  }

  return result;
}

llama_grammar *Llama::load_grammar(const std::string &grammar_text) {

  if (!grammar_text.empty()) {

    this->parsed_grammar = grammar_parser::parse(grammar_text.c_str());

    // will be empty (default) if there are parse errors
    if (parsed_grammar.rules.empty()) {
      return NULL;
    }

    fprintf(stderr, "\nGRAMMAR:\n");
    grammar_parser::print_grammar(stderr, parsed_grammar);
    fprintf(stderr, "\n");

    std::vector<const llama_grammar_element *> grammar_rules(
        parsed_grammar.c_rules());
    return llama_grammar_init(grammar_rules.data(), grammar_rules.size(),
                              parsed_grammar.symbol_ids.at("root"));
  }

  return NULL;
}
