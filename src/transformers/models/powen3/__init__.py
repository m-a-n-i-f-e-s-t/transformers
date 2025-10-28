# Copyright 2024 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2TokenizerFast
from .configuration_powen3 import Powen3Config
from .modeling_powen3 import Powen3ForCausalLM

AutoConfig.register("powen3", Powen3Config)
AutoModelForCausalLM.register(Powen3Config, Powen3ForCausalLM)

AutoTokenizer.register(Powen3Config, GPT2Tokenizer, GPT2TokenizerFast)

__all__ = ["Powen3Config", "Powen3ForCausalLM"]