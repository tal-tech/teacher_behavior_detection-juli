import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
base_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(base_path,'src/auto_text_classifier'))
from src.teacher_behavior_detection import detect

def find_guidance_text(data):
    for item in data['data']['text_result']:
        if item['label'] == '举例':
            return item['text']

if __name__ == "__main__":
    input_text = [
        
        {
            "text": "比如这个题就需要",
            "begin_time": 1326752,
            "end_time": 1332165
        },
        {
            "text": "这个选b是不是？",
            "begin_time": 1326752,
            "end_time": 1332165
        }
    ]

    # 测试用
    for i in range(1):
        result = detect(input_text, keywords_scene='qingqing')
        print(find_guidance_text(result))
        # print(json.dumps(result, indent=4, ensure_ascii=False))
