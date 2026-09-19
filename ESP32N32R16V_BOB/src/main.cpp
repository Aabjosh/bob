#include <Arduino.h>
#include <SPIFFS.h>
#include <esp_heap_caps.h>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

typedef struct {
    int index;
    float prob;
} ProbIndex;

typedef struct {
    char* str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex* sorted_vocab;
    int vocab_size;
    int max_token_length;
    unsigned char byte_pieces[256 * 2];
} Tokenizer;

typedef struct {
    float temperature;
    float topp;
    int vocab_size;
    ProbIndex* probindex;
    uint32_t rng_state;
} Sampler;

static int compare(const void* a, const void* b) {
    const ProbIndex* pa = (const ProbIndex*)a;
    const ProbIndex* pb = (const ProbIndex*)b;
    if (pa->prob < pb->prob) return 1;
    if (pa->prob > pb->prob) return -1;
    return 0;
}

static int compare_tokens(const void* a, const void* b) {
    const TokenIndex* ta = (const TokenIndex*)a;
    const TokenIndex* tb = (const TokenIndex*)b;
    return strcmp(ta->str, tb->str);
}

static int compare_lookup_tokens(const void* a, const void* b) {
    const TokenIndex* ta = (const TokenIndex*)a;
    const TokenIndex* tb = (const TokenIndex*)b;
    return strcmp(ta->str, tb->str);
}

static int str_lookup(const char* needle, TokenIndex* sorted_vocab, int vocab_size) {
    TokenIndex key;
    key.str = const_cast<char*>(needle);
    key.id = -1;
    TokenIndex* found = (TokenIndex*)bsearch(&key, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_lookup_tokens);
    if (found == nullptr) return -1;
    return found->id;
}

static float random_f32(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return ((*state) & 0xFFFFFFu) / 16777216.0f;
}

static void rmsnorm(float* o, float* x, float* weight, int size);
static void softmax(float* x, int size);
static void matmul(float* xout, float* x, float* w, int n, int d);

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(*x));

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq + l*dim*dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l*dim*kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l*dim*kv_dim, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l*dim*dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // classifier into logits
    matmul(s->logits, x, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

bool build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    File file = SPIFFS.open(tokenizer_path, FILE_READ);
    if (!file) {
        Serial.printf("couldn't load %s\n", tokenizer_path);
        return false;
    }

    if (file.read((uint8_t*)&t->max_token_length, sizeof(int)) != sizeof(int)) {
        Serial.println("failed read tokenizer header");
        file.close();
        return false;
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (file.read((uint8_t*)(t->vocab_scores + i), sizeof(float)) != sizeof(float)) {
            Serial.println("failed read vocab score");
            file.close();
            return false;
        }
        if (file.read((uint8_t*)&len, sizeof(int)) != sizeof(int)) {
            Serial.println("failed read token length");
            file.close();
            return false;
        }
        t->vocab[i] = (char*)malloc(len + 1);
        if (file.read((uint8_t*)t->vocab[i], len) != len) {
            Serial.println("failed read token text");
            file.close();
            return false;
        }
        t->vocab[i][len] = '\0';
    }
    file.close();
    return true;
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}


void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;

        if (i % 50 == 0) yield();
    }
  
   
}


bool malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = (float*)ps_calloc(p->dim, sizeof(float));
    s->xb = (float*)ps_calloc(p->dim, sizeof(float));
    s->xb2 = (float*)ps_calloc(p->dim, sizeof(float));
    s->hb = (float*)ps_calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)ps_calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)ps_calloc(p->dim, sizeof(float));
    s->key_cache = (float*)ps_calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)ps_calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*)ps_calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)ps_calloc(p->vocab_size, sizeof(float));
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        Serial.println("malloc failed!");
        return false;
    }
    return true;
}

Config config;
TransformerWeights weights;
RunState state;
Tokenizer tokenizer;
bool model_loaded = false;

static void print_memory(const char* stage) {
    Serial.printf("BOOT: %s, heap=%u, psram=%u\n", stage,
                  (unsigned int)ESP.getFreeHeap(),
                  (unsigned int)ESP.getFreePsram());
    Serial.flush();
}

static bool read_model_array(File& file, float** out, size_t count) {
    if (count == 0) {
        *out = nullptr;
        return true;
    }

    *out = (float*)ps_calloc(count, sizeof(float));
    if (*out == nullptr) {
        Serial.printf("Failed to allocate %u floats\n", (unsigned int)count);
        return false;
    }

    size_t bytes = count * sizeof(float);
    if ((size_t)file.read((uint8_t*)(*out), bytes) != bytes) {
        Serial.println("Failed to read model data block");
        return false;
    }
    return true;
}

static bool load_model_from_spiffs(const char* path) {
    File file = SPIFFS.open(path, FILE_READ);
    if (!file) {
        Serial.printf("Failed to open model file: %s\n", path);
        return false;
    }

    if (file.read((uint8_t*)&config, sizeof(Config)) != sizeof(Config)) {
        Serial.println("Failed to read model header");
        file.close();
        return false;
    }

    bool shared_weights = config.vocab_size > 0;
    config.vocab_size = abs(config.vocab_size);
    if (config.dim <= 0 || config.hidden_dim <= 0 || config.n_layers <= 0 ||
        config.n_heads <= 0 || config.n_kv_heads <= 0 || config.seq_len <= 0 ||
        config.vocab_size <= 0 || config.dim % config.n_heads != 0 ||
        config.n_heads % config.n_kv_heads != 0) {
        Serial.println("Invalid model configuration");
        file.close();
        return false;
    }

    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    size_t layer_count = (size_t)config.n_layers;
    size_t hidden_dim = (size_t)config.hidden_dim;

    if (!read_model_array(file, &weights.token_embedding_table, (size_t)config.vocab_size * (size_t)config.dim)) return false;
    if (!read_model_array(file, &weights.rms_att_weight, layer_count * config.dim)) return false;
    if (!read_model_array(file, &weights.wq, layer_count * (size_t)config.dim * (size_t)config.dim)) return false;
    if (!read_model_array(file, &weights.wk, layer_count * (size_t)config.dim * (size_t)kv_dim)) return false;
    if (!read_model_array(file, &weights.wv, layer_count * (size_t)config.dim * (size_t)kv_dim)) return false;
    if (!read_model_array(file, &weights.wo, layer_count * (size_t)config.dim * (size_t)config.dim)) return false;
    if (!read_model_array(file, &weights.rms_ffn_weight, layer_count * config.dim)) return false;
    if (!read_model_array(file, &weights.w1, layer_count * hidden_dim * config.dim)) return false;
    if (!read_model_array(file, &weights.w2, layer_count * config.dim * hidden_dim)) return false;
    if (!read_model_array(file, &weights.w3, layer_count * hidden_dim * config.dim)) return false;
    if (!read_model_array(file, &weights.rms_final_weight, config.dim)) return false;

    size_t rope_bytes = (size_t)config.seq_len * (config.dim / config.n_heads) / 2 * 2 * sizeof(float);
    if (file.position() + rope_bytes > file.size()) {
        Serial.println("Model is truncated before RoPE tables");
        file.close();
        return false;
    }
    file.seek(file.position() + rope_bytes);

    if (shared_weights) {
        weights.wcls = weights.token_embedding_table;
    } else if (!read_model_array(file, &weights.wcls, (size_t)config.vocab_size * (size_t)config.dim)) {
        file.close();
        return false;
    }

    file.close();
    Serial.printf("Model header OK: dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d seq=%d\n",
                  config.dim, config.hidden_dim, config.n_layers, config.n_heads, config.n_kv_heads,
                  config.vocab_size, config.seq_len);
    return true;
}

void setup() {
    Serial.begin(115200);

    Serial.println("BOOT: started");
    Serial.flush();
    print_memory("serial ready");

    if (!SPIFFS.begin(false)) {
        Serial.println("BOOT: SPIFFS Mount Failed. Filesystem was not formatted.");
        return;
    }
    Serial.printf("BOOT: SPIFFS mounted, total=%u used=%u\n",
                  (unsigned int)SPIFFS.totalBytes(),
                  (unsigned int)SPIFFS.usedBytes());
    print_memory("SPIFFS ready");

    if (!load_model_from_spiffs("/stories260K.bin")) {
        Serial.println("BOOT: Model load failed.");
        return;
    }
    Serial.println("BOOT: model loaded");
    print_memory("model loaded");

    if (!build_tokenizer(&tokenizer, "/tok512.bin", config.vocab_size)) {
        Serial.println("BOOT: Tokenizer load failed.");
        return;
    }
    Serial.println("BOOT: tokenizer loaded");
    print_memory("tokenizer loaded");

    if (!malloc_run_state(&state, &config)) {
        Serial.println("BOOT: runtime allocation failed.");
        return;
    }
    model_loaded = true;
    Serial.println("BOOT: Ready. Type a prompt and hit enter.");
    print_memory("ready");
}

void loop() {
    if (!model_loaded) {
        return;
    }

    if (Serial.available() > 0) {
        String prompt = Serial.readStringUntil('\n');
        prompt.trim();

        if (prompt.length() > 0) {
            Serial.printf("\nPrompt: %s\n", prompt.c_str());

            size_t prompt_capacity = prompt.length() + 3;
            char* prompt_buf = (char*)malloc(prompt_capacity);
            int* token_ids = (int*)malloc(prompt_capacity * sizeof(int));
            if (!prompt_buf || !token_ids) {
                Serial.println("Prompt allocation failed.");
                free(prompt_buf);
                free(token_ids);
                return;
            }
            prompt.toCharArray(prompt_buf, prompt_capacity);
            int n_tokens = 0;
            encode(&tokenizer, prompt_buf, 1, 0, token_ids, &n_tokens);

            if (n_tokens > config.seq_len) {
                Serial.printf("Prompt is too long (%d tokens, max %d).\n", n_tokens, config.seq_len);
                free(prompt_buf);
                free(token_ids);
                return;
            }

            Sampler sampler;
            sampler.temperature = 0.8f;
            sampler.topp = 0.9f;
            sampler.vocab_size = config.vocab_size;
            sampler.probindex = (ProbIndex*)ps_calloc(config.vocab_size, sizeof(ProbIndex));
            sampler.rng_state = (uint32_t)micros();
            if (!sampler.probindex) {
                Serial.println("Sampler allocation failed.");
                free(prompt_buf);
                free(token_ids);
                return;
            }

            int current_token = token_ids[0];
            int previous_token = 1;
            constexpr int max_new_tokens = 128;
            int total_steps = min(config.seq_len, n_tokens + max_new_tokens);
            for (int pos = 0; pos < total_steps; pos++) {
                Transformer model = { config, weights, state, 0, nullptr, 0 };
                float* logits = forward(&model, current_token, pos);
                int next = sample(&sampler, logits);
                if (next < 0 || next >= config.vocab_size) next = 0;

                if (pos >= n_tokens - 1) {
                    char* piece = decode(&tokenizer, previous_token, next);
                    Serial.print(piece);
                    previous_token = next;
                } else {
                    previous_token = token_ids[pos];
                }

                if (pos < n_tokens - 1) {
                    current_token = token_ids[pos + 1];
                } else {
                    current_token = next;
                }

                if (next == 2) {
                    break;
                }
                yield();
            }

            Serial.println("\n--- Done ---");
            free(sampler.probindex);
            free(prompt_buf);
            free(token_ids);
        }
    }
}