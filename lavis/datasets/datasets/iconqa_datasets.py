import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset

import torch
from PIL import Image


class IconQADataset(BaseDataset):
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
        choices = self._build_choices_string(ann["choices"])

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "choices": choices,
            "answer_idx": ann["answer_label"],
            "answers": ann["choices"][ann["answer_label"]],
            "weights": [1],
        }

    def collater(self, samples):
        image_list, text_input_list, answer_list, weight_list = [], [], [], []

        for sample in samples:
            image_list.append(sample["image"])

            # direct tuning with the original question
            text_input_list.append(sample["text_input"])
            answer_list.append(sample["answers"])
            weight_list += [1]

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "answer": answer_list,
            "weight": weight_list,
        }


class IconQAEvalDataset(IconQADataset):
    def collater(self, samples):
        (
            image_list,
            text_input_list,
            question_id_list,
            answer_list,
            direct_answer_list,
        ) = ([], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            text_input_list.append(
                (
                    sample["text_input"],
                    sample["choices"],
                )
            )
            question_id_list.append(sample["question_id"])
            answer_list.append(sample["answer_idx"])
            direct_answer_list.append(sample["answers"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            "answer_idx": answer_list,
            "direct_answer": direct_answer_list,
        }
