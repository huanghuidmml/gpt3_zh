from gpt3 import GPT3ForCausalLM
from transformers import BertTokenizerFast, GenerationConfig


model = GPT3ForCausalLM.from_pretrained("HuiHuang/gpt3-base-zh")
tokenizer = BertTokenizerFast.from_pretrained("HuiHuang/gpt3-base-zh")
generate_config = GenerationConfig(
    max_new_tokens=121, eos_token_id=tokenizer.sep_token_id,
    no_repeat_ngram_size=3, top_p=0.9, do_sample=True)
model.eval()

while True:
    user = input(">>>")
    model_inputs = tokenizer(
        user, padding=False, add_special_tokens=False, return_tensors='pt'
    )

    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask", None)
    generated_sequence = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generate_config
    )
    generated_sequence = generated_sequence[0]

    pred = tokenizer.decode(
        generated_sequence,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True)
    pred = pred.replace(" ", "")
    print(pred)
