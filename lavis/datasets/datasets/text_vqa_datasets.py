import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.vqa_datasets import VQADataset
import torch
from PIL import Image


class TextVQADataset(VQADataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, prompt):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths, prompt)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["images"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        ocr_tokens = f"[{','.join(ann['ocr_tokens'])}]"

        if self.prompt:
            text_input = self.prompt.format(ocr_tokens, question)
        else:
            text_input = question

        answer_weight = {}
        for answer in ann["answers"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answers"])
            else:
                answer_weight[answer] = 1 / len(ann["answers"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "image_id": ann["images_id"],
            "ocr_tokens": ocr_tokens,
            "text_input": text_input,
            "answers": answers,
            "weights": weights,
        }


class TextVQAEvalDataset(TextVQADataset):
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
                    sample["ocr_tokens"],
                    sample["text_input"],
                )
            )
            question_id_list.append(sample["image_id"])
            answer_list.append(sample["answers"])
        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": text_input_list,
            "question_id": question_id_list,
            "answers": answer_list,
        }
