import json

with open("test_vqa_result.json", "r") as f:
    res = json.load(f)


def parse_answer(answer):
    if len(answer) == 0:
        return ""
    return answer.strip("()")
    answer = answer.split()[0].strip("()").lower()
    return answer
    # if len(answer) > 1:
    #     return -1
    # return ord(answer.lower()) - ord('a')


org_pred = [parse_answer(d["org_pred_ans"]) for d in res]

print(org_pred)
