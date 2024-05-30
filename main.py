import pandas as pd
import json
from pathlib import Path
from pandas import json_normalize
import logging

import xml.etree.ElementTree as ET
import time
import argparse
import wandb

import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import transformers
import torch
import os
import leetcode
import json
import leetcode.auth
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0,'/home/dsshay/GNN_in_program_analysis/devign')

from devign.process import get_gnn_prediction
import pandas as pd

def find_file_with_substring(directory, substring):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if substring in file:
                return os.path.join(root, file)
    return None

def read_file_to_string(file_path):
    with open(file_path, 'r') as file:
        file_contents = file.read()
    return file_contents

def get_input_from_task(task: str, lang: str, dir, epoch):
    logging.info(f"Epoch: {epoch}, Lang:{lang}, Dir:{dir}")
    chat = [
        {"role": "system", "content": f"You are a helpful and honest code assistant expert in {lang.capitalize()}. Please, provide all answers to programming questions in {lang.capitalize()}."},
        {"role": "user", "content": "Write method as a single function. "+task},
    ]
    if int(epoch) > 0:
        for i in range(int(epoch)):
            direct = dir + f"/{i}/"
            file_path_model_output = find_file_with_substring(direct, f"Solution")
            if file_path_model_output:
                logging.info(f"Found model output: {file_path_model_output}")
            else:
                logging.error(
                    f"Not found model output to from directory:{direct}")
            chat.append({"role": "assistant", "content": read_file_to_string(file_path_model_output)})

            file_path_svace_output = find_file_with_substring(direct, f"svace_message")
            if file_path_svace_output:
                logging.info(f"Found svace output: {file_path_svace_output}")
            else:
                logging.error(
                    f"Not found svace output to from directory:{direct}")
            chat.append({"role": "user", "content": "correct program above with this feedback: Error:"+read_file_to_string(file_path_svace_output) + ".\n Write the resulting code."})
    logging.info(chat)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    return inputs


# Function to generate code for task
def process_task(task, lang, dir, epoch):
    # Send task to the language model
    if lang != "java" or lang != "python" or lang != "go" or lang != "kotlin" or lang != "C":
        Exception("Undefined lanuage of programming. Use only java, python, go, kotlin")

    question_dir = dir
    if epoch > 0:
        question_dir = os.path.dirname(os.path.dirname(dir))
        logging.info(f"Path question directory: {question_dir}")

    inputs = get_input_from_task(task, lang=lang, dir=question_dir, epoch=epoch)

    logging.info("Start generate code")
    output = model.generate(input_ids=inputs, max_new_tokens=2048)
    output = tokenizer.decode(output[0].to("cpu"), skip_special_tokens=True)
    output_file_path = os.path.join(dir, f"model_output.txt")
    with open(output_file_path, "w") as f:
        f.write(output)
    logging.info(f"Finished generate code, saved in {output_file_path}")
    return output


# Svace has problems analyzing programs in C
def svace_analyze(file, lang, epoch, dir):
    logging.info(f"File Name:{file}, Lang: {lang}, Directory: {dir}, Epoch: {epoch}")
    compiler_comand = ""
    result=""
    try:
        test = subprocess.run(f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace init", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True)
        logging.info(f"What happend? {test.stdout}")
        test = subprocess.run(f"pwd", shell=True, check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True)
        logging.info(f"Current Directory: {test.stdout}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error message: {e.stderr}")
        exit(1)

    if lang == "java":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build javac {file}"
    elif lang == "python":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build --python {file}"
    elif lang == "go":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build go build {file}"
    elif lang == "kotlin":
        compiler_comand = f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace build kotlinc {file}"
    else:
        Exception("Undefined lanuage of programming. Use only java, python, go, kotlin. Sensetive to capitalization")


    try:
        test = subprocess.run(compiler_comand, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)
        logging.info(f"Svace build out: {test.stdout} for file: {file}")
    except subprocess.CalledProcessError as e:
        logging.info(f"svace build: {test.stdout}")
        logging.error(f"Error executing command: {compiler_comand}")
        logging.error(f"Error message: {e.stderr}")
        if len(e.stderr)==0:
            result = "Write the full code with the correction."
        else:
            result = e.stderr
            result = result[:result.find("svace build: error:")]
    if len(result)==0:
        try:
            test = subprocess.run(f"cd {dir}; ~/GNN_in_program_analysis/svace-3.4.240117-x64-linux/bin/svace analyze", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True)
            logging.info(f"What happend? {test.stdout}")
            directory = dir + ".svace-dir/analyze-res"
            files = os.listdir(directory)
            svres_files = [file for file in files if file.endswith(".svres")]
            txt = [file for file in files if file.endswith(f"{epoch}.txt")]
            if len(txt) != 0:
                svace_an = read_file_to_string(directory+f"/{txt[0]}")
                lines = svace_an.strip().split("\n")
                try:
                    total_warnings = int(lines[0].split(":")[1].strip())
                    logging.info(f"Total warning={total_warnings} in epoch:{epoch}, question_id:{question_id}")
                    return 0
                except IndexError:
                    tree = ET.parse(directory + f"/{svres_files[0]}")
                    root = tree.getroot()
                    result = ""
                    for warn_info in root.findall(".//WarnInfo"):
                        line = warn_info.attrib.get("line")
                        warning_msg = warn_info.attrib.get("msg")
                        if warning_msg:
                            result += f"In Line {line}: {warning_msg}\n"
            else:
                logging.error("Not Found analyze result file.txt")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing command: svace analyze")
            logging.error(f"Error message: {e.stderr}")
            exit(1)

    output_file_path = os.path.join(dir, f"svace_message.txt")
    with open(output_file_path, "w") as f:
        f.write(result)
    logging.info(f"Finished Svace analyzing, result saved in {output_file_path}")
    return 1


def devign_analyze(func_text, lang): 
    predictions = get_gnn_prediction([func_text], None)
    print('Current predictions:', predictions)


# Function to submitting code to leetcode.com
def submit_to_leetcode(code,question_id, name_problem, api_instance, lang, dir, epoch):
    logging.info(f"Parametrs of submission: Question_id = {question_id}, name_problem = {name_problem}, lang = {lang}, epoch = {epoch}")
    if lang != "java" or lang != "python" or lang != "go" or lang != "kotlin":
        Exception("Undefined lanuage of programming. Use only java, python, go, kotlin. Sensetive to capitalization")

    submission = leetcode.Submission(
        judge_type="large", typed_code=code, question_id=question_id, test_mode=False, lang=lang
    )

    try: 
        submission_id = api_instance.problems_problem_submit_post(
            problem=name_problem, body=submission
        )
    except Exception as e:
        logging.error(f"Exception occurred: {e}. Leetcode submit failed. Maybe the problem is premium?")
        return None

    logging.info(f"The solution to the problem {name_problem} ({question_id}) submitted, the submission_id={submission_id}")

    time.sleep(10)

    submission_result = api_instance.submissions_detail_id_check_get(
        id=submission_id.submission_id
    )

    json_formatted_str = json.dumps(submission_result, indent=2)

    output_file_path = os.path.join(dir, f"result.json")
    with open(output_file_path, "w") as f:
        f.write(json_formatted_str)
    logging.info(
        f"Submission result saved in {output_file_path}")
    return submission_result


def load_data():
    if not os.path.exists('./data/leetcode-train.jsonl'):
        hf_hub_download(repo_id="greengerong/leetcode", filename="leetcode-train.jsonl", local_dir='./data/', local_dir_use_symlinks=False, repo_type="dataset")
    logging.info("Start download file")
    df = pd.read_json("./data/leetcode-train.jsonl", lines=True).drop(['python', 'javascript', 'java', 'c++'], axis=1)

    # df.difficulty.fillna(value="Medium", inplace=True)
    # train, test = train_test_split(df, stratify=df['difficulty'], random_state=42, test_size=0.2)
    # train, val = train_test_split(train, stratify=train['difficulty'], random_state=42, test_size=0.1)
    # logging.info(f"Train size f{train.shape}, Validate size {val.shape}, Test size {test.shape}")
    logging.info(f"Finished download file, dataframe size: {df.size}")
    # return train, val, test
    return df


def construct_leetcode_config(csrf_token, leetcode_session):
    configuration = leetcode.Configuration()

    configuration.api_key["x-csrftoken"] = csrf_token
    configuration.api_key["csrftoken"] = csrf_token
    configuration.api_key["LEETCODE_SESSION"] = leetcode_session
    configuration.api_key["Referer"] = "https://leetcode.com"
    configuration.debug = False

    api_instance = leetcode.DefaultApi(leetcode.ApiClient(configuration))

    graphql_request = leetcode.GraphqlQuery(
        query="""
        {
          user {
            username
            isCurrentUserPremium
          }
        }
            """,
        variables=leetcode.GraphqlQueryVariables(),
    )

    print(api_instance.graphql_post(body=graphql_request))
    return api_instance


if __name__ == "__main__":
    logging.basicConfig(
        filename='HISTORY.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger().setLevel(logging.INFO)

    

    # Get the next two values from your browser cookies
    leetcode_session = ""
    csrf_token = ""
    parser = argparse.ArgumentParser()

    # System arguments
    parser.add_argument("--dataset", default='leetcode-train.jsonl', type=str)
    parser.add_argument("--mode", default='test', type=str)
    parser.add_argument("--wandb_flag", default=False, type=str)
    parser.add_argument("--lang", default="java", type=str)
    parser.add_argument("--num_epochs", default=5, type=int)

    args = parser.parse_args()
    print(f"Arguments: {args}")

    if args.wandb_flag:
        wandb.init(project='gnn_in_pa', entity="gnn_in_pa", config=args, tags=["test"])

    model_id = ""
    if args.mode == 'test':
        model_id = "codellama/CodeLlama-7b-Instruct-hf"
    elif args.mode == 'prod':
        model_id = "codellama/CodeLlama-70b-Instruct-hf"
    else:
        print(f'The mode can be of two types: test or prod')
        Exception()

    df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(device)

    formats = args.lang
    if args.lang == "python":
        formats = "py"
    elif args.lang == "kotlin":
        formats = "kt"

    main_directory = f"./llm_predicts/{formats}"
    os.makedirs(f"./llm_predicts/{formats}", exist_ok=True)

    functions = []
    vulns = []

    api_instance = construct_leetcode_config(csrf_token,leetcode_session)
    print('Constructed leetcode config!')
    for row_index, row in df.iterrows():
        print('Analyzing new problem!')

        name_problem = row['slug']
        question_id = row['id']
        task = row['content']

        analyzer_feedback = None
        output_directory = main_directory + f"/{question_id}/"
        os.makedirs(output_directory, exist_ok=True)
        for epoch in range(args.num_epochs):
            output_directory = main_directory + f"/{question_id}/{epoch}/"
            os.makedirs(output_directory, exist_ok=True)

            output = process_task(task, lang=args.lang, dir=output_directory, epoch=epoch)
            
            logging.info("Start selecting output")
            start = '```'
            end = '```'
            index_code = output.find(end)
            if index_code != -1:
                output = output[index_code + len(start) + 1:]
                output = output[:output.find(start)]
            else:
                logging.error("It is not possible to correctly select the code from the model's response. Perhaps there is not enough response length?")
                break

            # logging.info(f"Selected output: {output}")
            # if args.lang == "java":
            #     output = "class Solution{\n" + output + "}"

            output_file_path = os.path.join(output_directory, f"Solution.{formats}")
            with open(output_file_path, "w") as f:
                f.write(output)

            logging.info(f"Finished selecting output, saved in {output_file_path}")
            logging.info(f"Output is {output}")

            # formatted_responce_leetcode = submit_to_leetcode(code=output, question_id=question_id, name_problem=name_problem, api_instance=api_instance, epoch=epoch, lang=args.lang, dir=output_directory)
            # exit(0)
            # if formatted_responce_leetcode == None:
            #    break

            # svace_flag = svace_analyze(file=f"Solution.{formats}", lang=args.lang, dir=output_directory, epoch=epoch)
            functions.append(output.replace('"', '\"'))

            if row_index >= 1 and row_index % 2 == 1:
                functions_current = functions[len(vulns):]
                devign_flag = get_gnn_prediction(functions_current)

                if len(devign_flag) != len(functions_current):
                    functions = functions[:len(vulns)]
                    break

                vulns.extend(devign_flag[-len(functions_current):])
                print(len(functions), len(vulns))

                # Save as pd dataframe to column
                pd_df = pd.DataFrame(
                {'func': functions,
                'vulns': vulns
                })
                pd_df.to_csv('gnn_vuln_csv', index=False)

                if devign_flag[-1] >= 0.5:
                    vuln_cnt = 0
                    for vuln_prob in vulns:
                        vuln_cnt += int(vuln_prob >= 0.5)

                    print(f'Item {str(row_index)} contains vulnerabilities! Already {vuln_cnt} vulnerable functions detected!')

            break