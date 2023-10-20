from LAVIS.lavis.datasets.builders import load_dataset
import torch
from LAVIS.lavis.models import load_model_and_preprocess
import numpy as np


device = torch.device("cuda")
model, vis_processors, txt_processors = load_model_and_preprocess(
    name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
)


vqa_raw_dataset = load_dataset("science_qa")
answer = []
gt = []
failed_count = 0
batch_size = 16

for i in range(0, len(vqa_raw_dataset["test"]), batch_size):
    questions = [
        data["question"] for data in vqa_raw_dataset["test"][i : i + batch_size]
    ]
    images = torch.stack(
        [
            vis_processors["eval"](data["image"])
            for data in vqa_raw_dataset["test"][i : i + batch_size]
        ],
        dim=0,
    ).to(device)
    text_input = [
        (data["context"], data["question"], data["choices"])
        for data in vqa_raw_dataset["test"][i : i + batch_size]
    ]
    prompt = "Context: {} Question: {} Options: {} Answer: The answer is "
    predict_answer = model.predict_answers(
        samples={"image": images, "text_input": text_input},
        prompt=prompt,
        inference_method="generate",
    )
    print(predict_answer)
# for data in vqa_raw_dataset["test"]:
# # for i in range(100):
# #     data = vqa_raw_dataset["test"][i]
#     question = data["question"]
#     image = vis_processors["eval"](data["image"]).unsqueeze(0).to(device)
#     text_input = (data["context"], data["question"], data["choices"])
#     prompt = "Context: {} Question: {} Options: {} Answer: The answer is "
#     predict_answer = model.predict_answers(samples={"image": image, "text_input": [text_input]}, prompt=prompt, inference_method="generate")
#     if len(predict_answer[0]) > 1:
#     #     for i, ch in enumerate(data["choices_list"]):
#     #         if ch.lower() == predict_answer[0].lower():
#     #             predict_answer = [chr(i + ord('a'))]
#     #             break
#     #         elif ch.lower() in predict_answer[0].lower():
#     #             predict_answer = [chr(i + ord('a'))]
#     #             break
#     #     else:
#         predict_answer = ['z']
#         failed_count += 1
#     answer += predict_answer
#     gt += data["answer_idx"]
# answer = np.array([ord(a) - ord('a') for a in answer])
# gt = np.array(gt)
# print(failed_count)
# print((np.array(answer) == np.array(gt)).mean())
