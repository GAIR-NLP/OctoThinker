# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union, Optional
import copy
import pandas as pd
from collections import defaultdict
import pdb

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F

cpt_base='''Question:
[PROMPT]
Answer:
'''

ua_base='''User:
[PROMPT]
Assistant:
'''

orz_base="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: You must put your answer inside \\boxed{} and Your final answer will be extracted automatically by the \\boxed{} tag.\n[PROMPT]\nAssistant:"

icl_base='''A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. 

User: What is the largest value of $x$ such that the expression \[\\dfrac{x+1}{8x^2-65x+8}\] is not defined?
Assistant: In this particular case, the fraction will be undefined only if its denominator is equal to zero. 
Because of this, we can ignore the numerator. 
We start by setting the binomial in the denominator equal to 0: 
\\begin{align*} 8x^2-65x+8=0
\\\\Rightarrow\\qquad (8x-1)(x-8)=0
\\end{align*} 
We find that the two possible values for $x$ are $\\frac18$ and $8$. 
Since the question asks for the largest value, the final solution is $\\boxed{8}$.

User: BoatWorks built 3 canoes in January of this year and then each subsequent calendar month they built twice the number of canoes they had built the previous month. How many total canoes were built by BoatWorks by the end of March of this year?
Assistant: The number of boats built is $3+3\cdot2+3\cdot2^2 = 3+6+12 = \\boxed{21}$.

User: Kevin Kangaroo begins hopping on a number line at 0. He wants to get to 1, but he can hop only $\\frac{1}{3}$ of the distance. Each hop tires him out so that he continues to hop $\\frac{1}{3}$ of the remaining distance. How far has he hopped after five hops? Express your answer as a common fraction.
Assistant: Kevin hops $1/3$ of the remaining distance with every hop. 
His first hop takes $1/3$ closer. 
For his second hop, he has $2/3$ left to travel, so he hops forward $(2/3)(1/3)$. 
For his third hop, he has $(2/3)^2$ left to travel, so he hops forward $(2/3)^2(1/3)$. 
In general, Kevin hops forward $(2/3)^{k-1}(1/3)$ on his $k$th hop. 
We want to find how far he has hopped after five hops. 
This is a finite geometric series with first term $1/3$, common ratio $2/3$, and five terms. 
Thus, Kevin has hopped $\\frac{\\frac{1}{3}\\left(1-\\left(\\frac{2}{3}\\right)^5\\right)}{1-\\frac{2}{3}} = \\boxed{\\frac{211}{243}}$.

User: What is the area of the region defined by the equation $x^2+y^2 - 7 = 4y-14x+3$?
Assistant: We rewrite the equation as $x^2 + 14x + y^2 - 4y = 10$ and then complete the square, 
resulting in  $(x+7)^2-49 + (y-2)^2-4=10$, 
or $(x+7)^2+(y-2)^2=63$. 
This is the equation of a circle with center $(-7, 2)$ and radius $\\sqrt{63},$ 
so the area of this region is $\\pi r^2 = \\boxed{63\pi}$.

User: You must put your answer inside \\boxed{} and Your final answer will be extracted automatically by the \\boxed{} tag.
[PROMPT]
Assistant:
'''



base_0221="""A conversation between user and assistant. The user asks a question, and the assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <thinking> </thinking> and <answer> </answer> tags, respectively, i.e., 
<thinking>
reasoning process here
</thinking>
<answer>
answer here
</answer>
<|endoftext|> (the end of the response of assistant). Remember to put the final answer between the <answer> </answer> tags in the format of \\boxed{{answer}}.
user
[PROMPT]
assistant
<thinking>"""


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 filter_overlong_prompts=False,
                 template_type='default',
                 split=None):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.filter_overlong_prompts = filter_overlong_prompts

        self.template_type=template_type
        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()
        self.split=split

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _init_prompt_profile(self):
        self.passratio = {self.dataframe.iloc[i][self.prompt_key][1]['content'].strip():[0] for i in range(len(self.dataframe))}
     
    def update_prompt_profile(self, prompts, rewards):
        batch_passratio=defaultdict(list)
        for prompt, reward in zip(prompts, rewards):
            batch_passratio[prompt.strip()].append(reward)
        for prompt, reward in batch_passratio.items():
            batch_passratio[prompt.strip()]=np.mean(reward)
        for prompt, reward in batch_passratio.items():
            try:
                self.passratio[prompt.strip()].append(reward)
            except:
                continue
 
    def update_dataset(self): 
        self.dataframe=self.dataframe[self.dataframe.apply(lambda x: self.passratio[x[self.prompt_key][1]['content'].strip()][-1]<=1.0, axis=1)]

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            if self.template_type=="default":
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length, axis=1)]
            elif self.template_type=="orz_base":
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(tokenizer.encode(orz_base.replace("[PROMPT]", doc[prompt_key][1]['content']))) <= self.max_prompt_length-20, axis=1)]
            elif self.template_type=="base_0221":
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(tokenizer.encode(base_0221.replace("[PROMPT]", doc[prompt_key][1]['content']))) <= self.max_prompt_length-20, axis=1)]
            elif self.template_type=="icl_base":
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(tokenizer.encode(icl_base.replace("[PROMPT]", doc[prompt_key][1]['content']))) <= self.max_prompt_length-20, axis=1)]
            elif self.template_type=="cpt_base":
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(tokenizer.encode(cpt_base.replace("[PROMPT]", doc[prompt_key][1]['content']))) <= self.max_prompt_length-20, axis=1)]
            elif self.template_type=="ua_base":
                self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(tokenizer.encode(ua_base.replace("[PROMPT]", doc[prompt_key][1]['content']))) <= self.max_prompt_length-20, axis=1)]
            print(f'filter dataset len: {len(self.dataframe)}')
        self._init_prompt_profile()

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()

        chat = row_dict.pop(self.prompt_key)

        if self.template_type=="default":
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        elif self.template_type=="orz_base":
            prompt_with_chat_template = orz_base.replace("[PROMPT]", chat[1]['content'])
        elif self.template_type=="base_0221":
            prompt_with_chat_template = base_0221.replace("[PROMPT]", chat[1]['content'])
        elif self.template_type=="cpt_base":
            prompt_with_chat_template = cpt_base.replace("[PROMPT]", chat[1]['content'])
        elif self.template_type=="ua_base":
            prompt_with_chat_template = ua_base.replace("[PROMPT]", chat[1]['content'])
        elif self.template_type=="icl_base":
            if self.split=="train":
                prompt_with_chat_template = icl_base.replace("[PROMPT]", chat[1]['content'])
            else:
                prompt_with_chat_template = orz_base.replace("[PROMPT]", chat[1]['content'])
            

        is_multi_modal = self.image_key in row_dict
        if is_multi_modal:  # expand image token
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(image) for image in row_dict.pop(self.image_key)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if is_multi_modal:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = chat.tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
