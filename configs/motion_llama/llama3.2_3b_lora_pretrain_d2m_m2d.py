_base_ = 'llama3.2_3b_pretrain_d2m_m2d.py'

model = dict(
    lora_config=dict(
        _delete_=True,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "down_proj", "up_proj"],
        r=64,
        lora_alpha=128,
        lora_dropout=0.05
    ),
    freeze_emb='ori'
)
