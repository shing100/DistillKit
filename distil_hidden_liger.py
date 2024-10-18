import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from accelerate import Accelerator
from liger_kernel.transformers import AutoLigerKernelForCausalLM, apply_liger_kernel_to_llama
import yaml

# Configuration
config = {
    "project_name": "distil-multilayer-liger",
    "dataset": {
        "name": "mlabonne/FineTome-100k",
        "split": "train",
        "num_samples": 1000,
        "seed": 42
    },
    "models": {
        "teacher": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "student": "meta-llama/Llama-3.1-8B-Instruct"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n",
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 2,
        "save_total_limit": 2,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.2,
        "lr_scheduler_type": "linear",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 1.0,
        "group_by_length": False
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    },
    "liger_config": {
        "rope": True,
        "swiglu": True, 
        "cross_entropy": False,
        "fused_linear_cross_entropy": True,
        "rms_norm": True
    }
}

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

accelerator = Accelerator()
device = accelerator.device

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
if config["dataset"].get("num_samples"):
    dataset = dataset.select(range(config["dataset"]["num_samples"]))
dataset = dataset.shuffle(seed=config["dataset"]["seed"])

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

def prepare_dataset(example):
    system = "You are a helpful assistant chatbot."
    conversations = example['conversations']
    
    message = [{"role": "system", "content": system}]
    
    for conversation in conversations:
        if conversation.get('from') == 'human':
            message.append({"role": "user", "content": conversation.get('value', '')})
        elif conversation.get('from') == 'gpt':
            message.append({"role": "assistant", "content": conversation.get('value', '')})
    
    student_text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    teacher_text = teacher_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    student_encodings = student_tokenizer(student_text, truncation=True, max_length=config["tokenizer"]["max_length"], padding='max_length')
    teacher_encodings = teacher_tokenizer(teacher_text, truncation=True, max_length=config["tokenizer"]["max_length"], padding='max_length')

    return {
        "input_ids": student_encodings["input_ids"],
        "attention_mask": student_encodings["attention_mask"],
        "teacher_input_ids": teacher_encodings["input_ids"],
        "teacher_attention_mask": teacher_encodings["attention_mask"],
    }

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(prepare_dataset, remove_columns=original_columns)

print("Dataset preparation complete. Loading models...")

# Apply Liger Kernel optimizations
apply_liger_kernel_to_llama(**config["liger_config"])

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16 if config["training"]["bf16"] else (torch.float16 if config["training"]["fp16"] else torch.float32)}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

# Use AutoLigerKernelForCausalLM for the student model
teacher_model = AutoLigerKernelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs).to(device)
student_model = AutoLigerKernelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs).to(device)

class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states

adaptation_layer = MultiLayerAdaptationLayer(
    student_model.config.hidden_size,
    teacher_model.config.hidden_size,
    student_model.config.num_hidden_layers,
    teacher_model.config.num_hidden_layers,
    dtype=torch.bfloat16
).to(device)

from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.remove_unused_columns = kwargs.pop('remove_unused_columns', None)
        self.max_seq_length = kwargs.get('max_seq_length', 1024)
        self.loss_fn = LigerFusedLinearCrossEntropyLoss()
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        labels = inputs["labels"]
        
        student_outputs = model(**student_inputs, labels=labels, output_hidden_states=True)
        
        # Use Liger's fused linear cross entropy loss
        if hasattr(model, 'get_output_embeddings'):
            output_weights = model.get_output_embeddings().weight
            logits = student_outputs.logits
            original_loss = self.loss_fn(output_weights, logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            original_loss = student_outputs.loss

        self.teacher_model = self.teacher_model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["teacher_input_ids"],
                "attention_mask": inputs["teacher_attention_mask"],
            }
            
            teacher_outputs = teacher_model(**teacher_inputs, output_hidden_states=True)

        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, inputs, original_loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss
        
    def distillation_loss(self, student_outputs, teacher_outputs, inputs, original_loss):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        self.adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]
            
            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}")

            student_probs = F.softmax(adapted_student_hidden_states[student_hidden] / config["distillation"]["temperature"], dim=-1)
            teacher_probs = F.softmax(teacher_hidden / config["distillation"]["temperature"], dim=-1)

            loss_kd = F.kl_div(
                F.log_softmax(adapted_student_hidden_states[student_hidden] / config["distillation"]["temperature"], dim=-1),
                teacher_probs,
                reduction='batchmean'
            ) * (config["distillation"]["temperature"] ** 2)

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / hidden_dim

        total_loss = config["distillation"]["alpha"] * scaled_loss_kd + (1 - config["distillation"]["alpha"]) * original_loss
        return total_loss

# Training arguments
training_arguments = TrainingArguments(
    **config["training"],
    remove_unused_columns=False,
)

# Create the custom SFT Trainer
trainer = CustomSFTTrainer(
    model=student_model,
    train_dataset=dataset,
    max_seq_length=config["tokenizer"]["max_length"],
    tokenizer=student_tokenizer,
    args=training_arguments,
    packing=config["training"].get("packing", False),
)

# Add these attributes to the trainer
trainer.teacher_model = teacher_model
trainer.adaptation_layer = adaptation_layer
trainer.student_tokenizer = student_tokenizer
trainer.teacher_tokenizer = teacher_tokenizer

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])

# Save the adaptation layer
torch.save(adaptation_layer.state_dict(), os.path.join(config["training"]["output_dir"], "adaptation_layer.pth"))
