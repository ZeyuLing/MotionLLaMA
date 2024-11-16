DEFAULT_CHAT_TEMPLATE = (  # LLAMA3.1 used template
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}"
    "{% set content = bos_token + content %}{% endif %}"
    "{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
)


DEFAULT_RESPONSE_TEMPLATE = '<|start_header_id|>assistant<|end_header_id|>\n\n'
