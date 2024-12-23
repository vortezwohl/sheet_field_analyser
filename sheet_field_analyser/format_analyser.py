import json
import logging
from enum import Enum
from pprint import pprint

import pandas as pd

from ceo import get_openai_model
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from test_input.function_data import data
from sheet_field_analyser import logger as log

load_dotenv()
log.setLevel(logging.DEBUG)


class Language(Enum):
    Chinese = 'zh-cn'
    English = 'en-us'


def output_dict_formatter(response: str):
    response = response[response.find('{'):response.rfind('}') + 1]
    return json.loads(response)


def output_list_formatter(response: str):
    response = response[response.find('['):response.rfind(']') + 1]
    return json.loads(response)


def format_analysis(sheet: str, lang: Language, llm: BaseChatModel) -> dict:
    prompt = json.dumps({
        'objective': '根据以下<sheet>所示表格，捕获每个表字段的特征，并为每个表字段生成能够全面总结其特征的详细描述，你的输出类型为<JSON>',
        'hint_for_entity_recognition': '某些字段中存在某些隐含的变量, 你需要准确识别或挖掘出这些隐含变量并列举其已知取值范围(选择范围, 已知值集), '
                                       '对于取值范围(选择范围, 已知值集)受限的变量, 你需要在该字段的描述中准确列出所有已知的取值, '
                                       '字段中的变量取值范围(选择范围, 已知值集)使用 "<variable_name> ∈ [value_1, value_2, value_n, ...]" 表示.'
                                       '隐含变量可能包括: 车辆状态、交通场景、交通参与者、事件、风险等级...',
        'sheet': sheet,
        'output_format': {
            'field_1': '<description_for_field_1>',
            'field_2': '<description_for_field_2>',
            'field_3': '<description_for_field_3>',
            'field_n': '<description_for_field_n>'
        },
        'output_language': lang.value,
        'hint_for_language': '你所输出的语言取决于<output_language>，请你给出对应语言的输出'
    }, ensure_ascii=False)
    log.debug(f'format_analysis_prompt: {prompt}')
    _result = output_dict_formatter(llm.invoke(prompt).content)
    log.debug(f'format_analysis_response: {_result}')
    return _result


def update_format_analysis(fields_description_1: dict, fields_description_2: dict, lang: Language, llm: BaseChatModel) -> dict:
    prompt = json.dumps({
        'objective': [
            {
                'step_1': '比较<fields_description_1>与<fields_description_2>之间的差别, '
                          '并逐个捕获字段中的隐含变量取值范围(选择范围, 已知值集), 隐含变量可能包括: 车辆状态、交通场景、交通参与者、事件、风险等级...',
                'outputs': 'differences(markdown_formatted): <differences_between_fields_description_1_and_2>'
            },
            {
                'step_2': '合并<fields_description_1>与<fields_description_2>之间的字段描述差异和'
                          '字段中的隐含变量取值范围(选择范围, 已知值集, 使用 "<variable_name> ∈ [value_1, value_2, value_n, ...]" 表示), '
                          '生成<new_fields_description>',
                'hint_for_step_2': '除了字段描述差异, 字段内的隐含变量的取值范围(选择范围, 已知值集)的合并也同样重要',
                'outputs': 'new_fields_description: ```json{...}```'
            }
        ],
        'fields_description_1': fields_description_1,
        'fields_description_2': fields_description_2,
        'output_format': 'differences(markdown_formatted): <differences_between_fields_description_1_and_2>\n'
                         'new_fields_description: ```json' + str({
                            'field_1': '<description_for_field_1>',
                            'field_2': '<description_for_field_2>',
                            'field_3': '<description_for_field_3>',
                            'field_n': '<description_for_field_n>'
        }) + '```',
        'output_language': lang.value,
        'hint_for_language': '你所输出的语言取决于<output_language>所规定的语言，请你给出对应语言的输出'
    }, ensure_ascii=False)
    log.debug(f'update_format_analysis_prompt: {prompt}')
    content = llm.invoke(prompt).content
    print('迭代:', content)
    _result = output_dict_formatter(content)
    log.debug(f'update_format_analysis_response: {_result}')
    return _result


def peek_format_from_xls(
        file: str,
        lang: Language = Language.English,
        batch_size: int = 16,
        llm: BaseChatModel = get_openai_model()
) -> dict:
    if 'xls' not in file[file.rfind('.'):]:
        raise ValueError('file is not a .xls sheet.')
    df = pd.read_excel(file)
    field_dict = df.iloc[0].to_dict()
    fields = []
    one_shot_example = [field_dict]
    for k in field_dict.keys():
        fields.append(k)
    frame = []
    for _, _series in df.iterrows():
        frame.append(_series.to_dict())
    field_descr = format_analysis(str(frame[:batch_size]), lang=lang, llm=llm)
    frame = frame[batch_size:]
    while len(frame) > 0:
        new_field_descr = format_analysis(str(frame[:batch_size]), lang=lang, llm=llm)
        field_descr = update_format_analysis(new_field_descr, field_descr, lang=lang, llm=llm)
        frame = frame[batch_size:]
    return {
        'fields': fields,
        'field_description': field_descr,
        'output_example': one_shot_example
    }


def generate(context: str, output_descr: dict, llm: BaseChatModel) -> dict:
    prompt = json.dumps({
        'context': context,
        'objective': '你需要依据<context>中给出的参考资料, '
                     '遵循<output_description.field_description>给出的规则, '
                     '对<output_description.fields>进行填充生成.',
        'output_format': 'A list of JSON objects describing the functional disabilities analysis',
        'output_description': {
            'fields': output_descr['fields'],
            'field_description': output_descr['field_description']
        },
        'output_example': output_descr['output_example']
    }, ensure_ascii=False)
    log.debug(f'generate_prompt: {prompt}')
    _result = output_list_formatter(llm.invoke(prompt).content)
    log.debug(f'generate_response: {_result}')
    return _result


if __name__ == '__main__':
    _data = []
    for _test_data in data:
        _data.append(str(_test_data))
    _format = peek_format_from_xls('../static/hazard_analysis.xlsx', lang=Language.Chinese)
    pprint(_format)
    # result = generate(str(_data), output_descr=_format, llm=get_openai_model())
    # pprint(result)
