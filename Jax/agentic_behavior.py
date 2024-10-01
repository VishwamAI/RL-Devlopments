# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import List, Dict, Any, Callable, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from abc import ABC, abstractmethod
from NeuroFlex.utils.utils import tokenize_text, get_activation_function

class AgenticBehavior(ABC):
    @abstractmethod
    def zero_shot(self, prompt: str) -> str:
        pass

    @abstractmethod
    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        pass

    @abstractmethod
    def chain_of_thought(self, prompt: str) -> str:
        pass

    @abstractmethod
    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        pass

    @abstractmethod
    def self_correct(self, output: str) -> str:
        pass

    @abstractmethod
    def self_update(self, feedback: str) -> None:
        pass

    @abstractmethod
    def plan_and_execute(self, task: str) -> Tuple[List[str], str]:
        pass

    @abstractmethod
    def multi_agent_collaboration(self, task: str, other_agents: List['AgenticBehavior']) -> str:
        pass

class BaseAgent(AgenticBehavior):
    def __init__(self, model: nn.Module, tokenizer: Callable[[str], List[int]], vocab_size: int):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.optimizer = optax.adam(learning_rate=1e-4)
        self.opt_state = self.optimizer.init(self.model.params)

    def _encode_input(self, tokens: List[int]) -> jnp.ndarray:
        return jax.nn.one_hot(jnp.array(tokens), self.vocab_size)

    def _decode_output(self, output: jnp.ndarray) -> str:
        return " ".join([str(int(jnp.argmax(x))) for x in output])

    def self_correct(self, output: str) -> str:
        correction_prompt = f"The previous output was:\n{output}\n\nPlease review and correct any errors in the above output:"
        tokens = self.tokenizer(correction_prompt)
        encoded_input = self._encode_input(tokens)
        corrected_output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(corrected_output)

    def self_update(self, feedback: str) -> None:
        def loss_fn(params):
            tokens = self.tokenizer(feedback)
            encoded_input = self._encode_input(tokens)
            output = self.model.apply({'params': params}, encoded_input)
            return -jnp.mean(jnp.log(output[jnp.arange(len(tokens)), tokens]))

        grads = jax.grad(loss_fn)(self.model.params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model.params = optax.apply_updates(self.model.params, updates)

    def plan_and_execute(self, task: str) -> Tuple[List[str], str]:
        plan_prompt = f"Create a step-by-step plan to accomplish the following task:\n{task}"
        plan_tokens = self.tokenizer(plan_prompt)
        plan_input = self._encode_input(plan_tokens)
        plan_output = self.model.apply({'params': self.model.params}, plan_input)
        plan = self._decode_output(plan_output).split('\n')

        result = ""
        for step in plan:
            step_prompt = f"Execute the following step: {step}"
            step_tokens = self.tokenizer(step_prompt)
            step_input = self._encode_input(step_tokens)
            step_output = self.model.apply({'params': self.model.params}, step_input)
            result += self._decode_output(step_output) + "\n"

        return plan, result.strip()

    def multi_agent_collaboration(self, task: str, other_agents: List['AgenticBehavior']) -> str:
        results = [self.zero_shot(task)]
        for agent in other_agents:
            results.append(agent.zero_shot(task))

        collaboration_prompt = f"Task: {task}\n\nAgent outputs:\n" + "\n".join(results) + "\n\nSynthesize a final answer based on all agent outputs:"
        collab_tokens = self.tokenizer(collaboration_prompt)
        collab_input = self._encode_input(collab_tokens)
        collab_output = self.model.apply({'params': self.model.params}, collab_input)
        return self._decode_output(collab_output)

class NeuroFlexAgenticBehavior(BaseAgent):
    def zero_shot(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        context = self._format_examples(examples) + "\n" + prompt
        tokens = self.tokenizer(context)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

    def chain_of_thought(self, prompt: str) -> str:
        cot_prompt = f"Let's approach this step-by-step:\n1) {prompt}\n2) "
        tokens = self.tokenizer(cot_prompt)
        encoded_input = self._encode_input(tokens)

        thoughts = []
        for _ in range(5):  # Generate up to 5 steps
            output = self.model.apply({'params': self.model.params}, encoded_input)
            step = self._decode_output(output)
            thoughts.append(step)
            step_tokens = self.tokenizer(step)
            encoded_input = jnp.concatenate([encoded_input, self._encode_input(step_tokens)])

        return "\n".join(thoughts)

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        full_prompt = f"{meta_prompt}\n\nTask: {prompt}"
        tokens = self.tokenizer(full_prompt)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        formatted = ""
        for example in examples:
            formatted += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        return formatted.strip()

# Helper function to create a NeuroFlexAgenticBehavior instance
def create_neuroflex_agentic_behavior(model: nn.Module, tokenizer: Callable[[str], List[int]], vocab_size: int) -> NeuroFlexAgenticBehavior:
    return NeuroFlexAgenticBehavior(model, tokenizer, vocab_size)

class ZeroShotAgent(BaseAgent):
    def zero_shot(self, prompt: str) -> str:
        tokens = self.tokenizer(prompt)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        raise NotImplementedError("ZeroShotAgent does not support few-shot learning")

    def chain_of_thought(self, prompt: str) -> str:
        raise NotImplementedError("ZeroShotAgent does not support chain-of-thought reasoning")

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        raise NotImplementedError("ZeroShotAgent does not support meta-prompting")

class FewShotAgent(BaseAgent):
    def zero_shot(self, prompt: str) -> str:
        raise NotImplementedError("FewShotAgent does not support zero-shot learning")

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        context = self._format_examples(examples) + "\n" + prompt
        tokens = self.tokenizer(context)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

    def chain_of_thought(self, prompt: str) -> str:
        raise NotImplementedError("FewShotAgent does not support chain-of-thought reasoning")

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        raise NotImplementedError("FewShotAgent does not support meta-prompting")

    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        formatted = ""
        for example in examples:
            formatted += f"Input: {example['input']}\nOutput: {example['output']}\n\n"
        return formatted.strip()

class ChainOfThoughtAgent(BaseAgent):
    def zero_shot(self, prompt: str) -> str:
        raise NotImplementedError("ChainOfThoughtAgent does not support zero-shot learning")

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        raise NotImplementedError("ChainOfThoughtAgent does not support few-shot learning")

    def chain_of_thought(self, prompt: str) -> str:
        cot_prompt = f"Let's approach this step-by-step:\n1) {prompt}\n2) "
        tokens = self.tokenizer(cot_prompt)
        encoded_input = self._encode_input(tokens)

        thoughts = []
        for _ in range(5):  # Generate up to 5 steps
            output = self.model.apply({'params': self.model.params}, encoded_input)
            step = self._decode_output(output)
            thoughts.append(step)
            step_tokens = self.tokenizer(step)
            encoded_input = jnp.concatenate([encoded_input, self._encode_input(step_tokens)])

        return "\n".join(thoughts)

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        raise NotImplementedError("ChainOfThoughtAgent does not support meta-prompting")

class MetaPromptingAgent(BaseAgent):
    def zero_shot(self, prompt: str) -> str:
        raise NotImplementedError("MetaPromptingAgent does not support zero-shot learning")

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        raise NotImplementedError("MetaPromptingAgent does not support few-shot learning")

    def chain_of_thought(self, prompt: str) -> str:
        raise NotImplementedError("MetaPromptingAgent does not support chain-of-thought reasoning")

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        full_prompt = f"{meta_prompt}\n\nTask: {prompt}"
        tokens = self.tokenizer(full_prompt)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

class SelfConsistencyAgent(BaseAgent):
    def __init__(self, model: nn.Module, tokenizer: Callable[[str], List[int]], vocab_size: int, num_samples: int = 5):
        super().__init__(model, tokenizer, vocab_size)
        self.num_samples = num_samples

    def generate_samples(self, prompt: str) -> List[str]:
        samples = []
        for _ in range(self.num_samples):
            tokens = self.tokenizer(prompt)
            encoded_input = self._encode_input(tokens)
            output = self.model.apply({'params': self.model.params}, encoded_input)
            samples.append(self._decode_output(output))
        return samples

    def select_most_consistent(self, samples: List[str]) -> str:
        from collections import Counter
        return Counter(samples).most_common(1)[0][0]

    def zero_shot(self, prompt: str) -> str:
        samples = self.generate_samples(prompt)
        return self.select_most_consistent(samples)

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        raise NotImplementedError("SelfConsistencyAgent does not support few-shot learning")

    def chain_of_thought(self, prompt: str) -> str:
        raise NotImplementedError("SelfConsistencyAgent does not support chain-of-thought reasoning")

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        raise NotImplementedError("SelfConsistencyAgent does not support meta-prompting")

class GenerateKnowledgePromptingAgent(BaseAgent):
    def __init__(self, model: nn.Module, tokenizer: Callable[[str], List[int]], vocab_size: int, knowledge_base: Dict[str, str]):
        super().__init__(model, tokenizer, vocab_size)
        self.knowledge_base = knowledge_base

    def generate_knowledge(self, prompt: str) -> str:
        relevant_knowledge = []
        for key, value in self.knowledge_base.items():
            if key.lower() in prompt.lower():
                relevant_knowledge.append(value)
        return " ".join(relevant_knowledge)

    def integrate_knowledge(self, prompt: str, knowledge: str) -> str:
        return f"Given the following knowledge: {knowledge}\n\nAnswer the question: {prompt}"

    def zero_shot(self, prompt: str) -> str:
        knowledge = self.generate_knowledge(prompt)
        integrated_prompt = self.integrate_knowledge(prompt, knowledge)
        tokens = self.tokenizer(integrated_prompt)
        encoded_input = self._encode_input(tokens)
        output = self.model.apply({'params': self.model.params}, encoded_input)
        return self._decode_output(output)

    def few_shot(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        raise NotImplementedError("GenerateKnowledgePromptingAgent does not support few-shot learning")

    def chain_of_thought(self, prompt: str) -> str:
        raise NotImplementedError("GenerateKnowledgePromptingAgent does not support chain-of-thought reasoning")

    def meta_prompting(self, prompt: str, meta_prompt: str) -> str:
        raise NotImplementedError("GenerateKnowledgePromptingAgent does not support meta-prompting")