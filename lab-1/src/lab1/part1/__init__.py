from transformers.integrations import sdpa_attention

sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
