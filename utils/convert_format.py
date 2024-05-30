"""
Этот файл конвертирует датасет, используемый для обучения в данной работе, в формат,
пригодный для использования выбранной реализацией модели Devign (т.е. который можно отдать в обработку Joern)
"""
import pandas as pd
import json

LINES_NUMBER = 38
FILES_TO_READ = 188636

VULNERABLE_FUNCTIONS = 10000
SAFE_FUNCTIONS = 25000

vulnerable_cnt = 0
safe_cnt = 0

with open('path_to_json_dataset_of_vuln_code', 'r') as f:
    next(f) # Skip first line

    with open('path_where_new_json_generates', 'w') as f_w:
        f_w.write('[')

        for file_id in range(FILES_TO_READ):
            if file_id % 1000 == 0:
                print('Now on file:', file_id, 'Safe:', safe_cnt, 'Vulnerable:', vulnerable_cnt)

            current_file = ""
            for line_id in range(LINES_NUMBER):
                current_line = next(f)
                if line_id == 0:
                    current_line = current_line.split(':')[1]
                elif line_id == LINES_NUMBER - 1:
                    current_line = current_line[:-2]
                current_file += current_line
            data = json.loads(current_file)

            data_cut = {}
            data_cut['commit_id'] = data['commit_id']
            data_cut['project'] = 'Custom'

            if data['vul'] == '0':
                if safe_cnt >= SAFE_FUNCTIONS:
                    continue

                safe_cnt += 1

                data_cut['func'] = data['func_before']
                data_cut['target'] = 0
            elif data['vul'] == '1':
                if vulnerable_cnt >= VULNERABLE_FUNCTIONS:
                    continue

                vulnerable_cnt += 1

                data_cut['func'] = data['func_before']
                data_cut['target'] = 1
            
            json.dump(data_cut, f_w)
            
            if safe_cnt >= SAFE_FUNCTIONS and vulnerable_cnt >= VULNERABLE_FUNCTIONS:
                break
            if file_id != FILES_TO_READ - 1:
                f_w.write(', ')

        f_w.write(']')

print('Finished with:')
print('Total functions:', FILES_TO_READ)
print('Vulnerable:', vulnerable_cnt)
print('Safe:', safe_cnt)