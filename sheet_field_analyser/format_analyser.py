import json
import logging
from enum import Enum
from pprint import pprint

import pandas as pd

from ceo import get_openai_model
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from test_input.function_data import data

load_dotenv()
log = logging.getLogger('sheet_field_analyser')
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
        'hint_for_entity_recognition': '某些字段中存在某些实体, 你需要识别出这些实体并列举实体的已知取值范围(选择范围, 已知值集), '
                                       '对于取值范围(选择范围, 已知值集)受限的实体, 你需要在该字段的描述中准确列出所有已知的取值, '
                                       '字段中的实体取值范围(选择范围, 已知值集)使用 "<entity_name> ∈ [value_1, value_2, value_n, ...]" 表示.'
                                       '实体可能包括: 状态、场景、角色、事件、风险等级...',
        'hint': '某些字段中存在某些特定域值, 你需要关注特定域值的取值范围(选择范围, 已知值集), 对于取值范围(选择范围, 已知值集)受限的特定域值, '
                '你需要在该字段的描述中准确列出所有特定域值的已知取值, '
                '字段中特定域值的取值范围(选择范围, 已知值集)使用 "<variable_name> ∈ [value_1, value_2, value_n, ...]" 表示',
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
                          '并分别捕获字段中的实体取值范围(选择范围, 已知值集), 实体可能包括: 状态、场景、角色、事件、风险等级...',
                'outputs': 'differences(markdown_formatted): <differences_between_fields_description_1_and_2>'
            },
            {
                'step_2': '合并<fields_description_1>与<fields_description_2>之间的字段描述差异和'
                          '字段中的实体取值范围(选择范围, 已知值集), 生成<new_fields_description>',
                'hint_for_step_2': '除了字段描述差异, 字段取值范围(选择范围, 已知值集)的合并也同样重要, '
                                   '字段中的实体或特定域值的取值范围(选择范围, 已知值集)'
                                   '使用 "<entity/variable_name> ∈ [value_1, value_2, value_n, ...]" 表示',
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
    # _test_data = test_data[:2]
    _test_data = [{'功能ID': 'FUN_0', '功能描述': '随速助力: 根据驾驶员输入的方向盘扭矩大小和车速大小，ECU计算目标助力电流以控制电机运行，实现基本助力功能。', 'HAZOP引导词': 'More', '失效ID': 'FUN_0_FM01', '功能失效描述': '在车辆低速行驶时，由于传感器误读或软件错误，随速助力功能误认为车辆处于高速状态而提供过大的助力。', '失效的影响': '可能导致驾驶员感觉转向过于轻便，影响对车辆的控制，特别是在需要精确操控的情况下，如停车或避障，增加了发生事故的风险。', '整车危害行为': '非预期转向'}, {'功能ID': 'FUN_1', '功能描述': '主动回正: 通过方向盘转角信号主动拉方向盘回中心，提高方向盘返回功能，并且方向盘回正速度可控。', 'HAZOP引导词': 'Less', '失效ID': 'FUN_1_FM01', '功能失效描述': '在高速行驶时，主动回正功能未能提供足够的回正扭矩，导致方向盘无法有效回正。', '失效的影响': '驾驶员可能需要额外的力量来手动回正方向盘，增加驾驶疲劳，降低驾驶安全性。', '整车危害行为': '转向能力丧失'}, {'功能ID': 'FUN_2', '功能描述': '软止点保护: 在驾驶员将方向盘转到极限位置时，降低驱动电流输出，减少助力，保护电机与机械结构。', 'HAZOP引导词': 'Omission', '失效ID': 'FUN_2_FM01', '功能失效描述': '软止点保护功能未能在方向盘达到极限位置时及时降低助力，导致电机过载。', '失效的影响': '可能导致电机过热或损坏，影响系统的长期可靠性和安全性。', '整车危害行为': '转向卡滞'}]
    _data = []
    for _test_data in data:
        _data.append(str(_test_data))
    _format = peek_format_from_xls('../static/hazard_analysis.xlsx', lang=Language.Chinese)
    pprint(_format)
    # result = generate(str(_data), output_descr=_format, llm=get_openai_model())
    # pprint(result)
