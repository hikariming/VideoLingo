import os, sys, json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from threading import Lock
import json_repair
import json 
from openai import OpenAI
import time
from requests.exceptions import RequestException

LOG_FOLDER = 'output/gpt_log'
LOCK = Lock()

def save_log(model, prompt, response, log_title = 'default', message = None):
    os.makedirs(LOG_FOLDER, exist_ok=True)
    log_data = {
        "model": model,
        "prompt": prompt,
        "response": response,
        "message": message
    }
    log_file = os.path.join(LOG_FOLDER, f"{log_title}.json")
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_data)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
        
def check_ask_gpt_history(prompt, model, log_title):
    # check if the prompt has been asked before
    if not os.path.exists(LOG_FOLDER):
        return False
    file_path = os.path.join(LOG_FOLDER, f"{log_title}.json")
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                if item["prompt"] == prompt and item["model"] == model:
                    return item["response"]
    return False

def ask_gpt(prompt, response_json=True, valid_def=None, log_title='default'):
    from config import MODEL, API_KEY, BASE_URL, llm_support_json
    with LOCK:
        history_response = check_ask_gpt_history(prompt, MODEL, log_title)
        if history_response:
            return history_response
    
    if not API_KEY:
        raise ValueError(f"⚠️API_KEY is missing")
    
    messages = [{"role": "user", "content": prompt}]
    
    base_url = BASE_URL.strip('/') + '/v1' if 'v1' not in BASE_URL else BASE_URL
    client = OpenAI(api_key=API_KEY, base_url=base_url)
    response_format = {"type": "json_object"} if response_json and MODEL in llm_support_json else None

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                response_format=response_format,
                timeout=150 #! set timeout
            )
            
            if response_json:
                try:
                    response_data = json_repair.loads(response.choices[0].message.content)
                    
                    # check if the response is valid, otherwise save the log and raise error and retry
                    if valid_def:
                        valid_response = valid_def(response_data)
                        if valid_response['status'] != 'success':
                            save_log(MODEL, prompt, response_data, log_title="error", message=valid_response['message'])
                            raise ValueError(f"❎ API response error: {valid_response['message']}")
                        
                    break  # Successfully accessed and parsed, break the loop
                except Exception as e:
                    response_data = response.choices[0].message.content
                    print(f"❎ json_repair parsing failed. Retrying: '''{response_data}'''")
                    save_log(MODEL, prompt, response_data, log_title="error", message=f"json_repair parsing failed.")
                    
                    # 新增：请求 AI 重新解析有问题的 JSON
                    repair_prompt = f"""
下面的JSON中有1-2个对象存在问题导致无法解析。请帮助重新解析并输出正确的JSON。
要求如下:
1. 所有的key必须是英文
2. 不要使用双引号包裹key和value
3. 保持原有的缩进结构
4. 只修复有问题的部分,保留其他正确的内容

原始JSON:
{response_data}

请直接输出修复后的JSON,无需其他解释。
"""
                    repaired_response = ask_gpt(repair_prompt, response_json=True, log_title="json_repair")
                    
                    try:
                        response_data = json.loads(repaired_response)
                        print("✅ JSON successfully repaired by AI.")
                        break
                    except json.JSONDecodeError:
                        print("❎ AI repair failed. Continuing with original error.")
                    
                    if attempt == max_retries - 1:
                        raise Exception(f"JSON parsing still failed after {max_retries} attempts and AI repair: {e}")
            else:
                response_data = response.choices[0].message.content
                break  # Non-JSON format, break the loop directly
                
        except Exception as e:
            if attempt < max_retries - 1:
                if isinstance(e, RequestException):
                    print(f"Request error: {e}. Retrying ({attempt + 1}/{max_retries})...")
                else:
                    print(f"Unexpected error occurred: {e}\nRetrying...")
                time.sleep(2)
            else:
                raise Exception(f"Still failed after {max_retries} attempts: {e}")
    with LOCK:
        if log_title != 'None':
            save_log(MODEL, prompt, response_data, log_title=log_title)

    return response_data

# test
if __name__ == '__main__':
    print(ask_gpt('hi there hey response in json format, just return 200.' , response_json=True, log_title=None))