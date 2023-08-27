
bool ignore_eos                         # ignore end of stream token and continue generating (implies --logit-bias 2-inf)
llama_msgs/LogitBiasArray logit_bias    # logit bias for specific tokens

float32 temp                            # temperature

int32 top_k                             # top-k sampling (0.0 = disabled)
float32 top_p                           # top-p sampling (1.0 = disabled)
float32 tfs_z                           # tail free sampling, parameter z (1.0 = disabled)
float32 typical_p                       # locally typical sampling, parameter p (1.0 = disabled)

int32 repeat_last_n                     # last n tokens consider for penalize (0 = disable penalty, -1 = context size)
float32 repeat_penalty                  # penalize repeat sequence of tokens (1.0 = disabled)
float32 presence_penalty                # repeat alpha presence penalty (0.0 = disabled)
float32 frequency_penalty               # repeat alpha frequency penalty (0.0 = disable)

int32 mirostat                          # Mirostart sampling (0 = disabled, 1 = mirostat, 2 = mirostat 2.0)
float32 mirostat_eta                    # Mirostat learning rate, parameter eta
float32 mirostat_tau                    # Mirostat target entropy, parameter tau

bool penalize_nl                        # consider newlines as a repeatable token
int32 n_probs                           # if greater than 0, output the probabilities of top n_probs tokens

string grammar                          # optional BNF-like grammar to constrain sampling