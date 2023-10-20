import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset

import torch
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "hint": ann["hint"],
                "question": ann["question"],
                "choices": ann["choices"],
                "correct_choice": ann["choices"][ann["answer"]],
                "image": sample["image"],
            }
        )


class ScienceQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, prompt):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, prompt)

    def _build_choices_string(self, choices):
        return " ".join(
            [f'({chr(i + ord("A"))}) {choice}' for i, choice in enumerate(choices)]
        )

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        context = self.text_processor(ann["hint"])
        choices = self._build_choices_string(ann["choices"])

        return {
            "image": image,
            "question": question,
            "question_id": ann["question_id"],
            "context": context,
            "choices": choices,
            "answer_idx": [ann["answer"]],
            "weights": [1],
        }

    def collater(self, samples):
        image_list, text_input_list, answer_list, weight_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(
                self.prompt.format(
                    sample["context"],
                    sample["question"],
                    sample["choices"],
                )
            )
            answer_list.append(f'({chr(ord("A") + sample["answer_idx"][0])})')
            weight_list += [1]

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "answer": answer_list,
            "weight": weight_list,
        }


class ScienceQAEvalDataset(ScienceQADataset, __DisplMixin):
    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            answer_list,
        ) = ([], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(
                (
                    sample["context"],
                    sample["question"],
                    sample["choices"],
                )
            )
            question_id_list.append(sample["question_id"])
            answer_list.append(sample["answer_idx"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            "answer_idx": answer_list,
        }
