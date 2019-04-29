import json
import os
import glob


def load_jsonl(file_path):
    data = []
    for line in open(file_path):
        entry = json.loads(line)
        data.append(entry)

    return data


def dump_data(example_file, table_file):
    examples = load_jsonl(example_file)
    tables = load_jsonl(table_file)
    tables = {x['name']: x for x in tables}

    output_examples = []
    for example in examples:
        question = example['question']
        table_id = example['context']
        table = tables[table_id]

        columns = []
        for column in table['props']:
            column_name = column[len('r.'):]
            type_pos = column_name.rfind('-')
            column_name = column_name[:type_pos]
            column_name = column_name.replace('-', ' ').replace('_', ' ')

            type_string = column[column.rfind('-') + 1:]

            if type_string == 'string':
                type_string = 'text'
            elif type_string.startswith('num') or type_string.startswith('date'):
                type_string = 'real'
            else:
                type_string = 'text'

            sample_value = None
            for row_id, row in table['kg'].items():
                if column in row and isinstance(row[column], list) and row[column][0] is not None:
                    sample_value = row[column][0]
                    break

            column_entry = {'name': column_name,
                            'type': type_string,
                            'sample_value': sample_value,
                            'is_target': False}
            columns.append(column_entry)

        output_examples.append({
            'guid': example['id'],
            'question': question,
            'columns': columns
        })

    return output_examples


def load_relation_prediction_results_to_examples(example_file, relation_prediction_result_file):
    examples = load_jsonl(example_file)
    rel_pred_results = json.load(open(relation_prediction_result_file))

    def _get_matched_props(_column_name, _prop_features):
        _column_name = _column_name.replace(' ', '_')
        _column_name = 'r.' + _column_name + '-'

        _matched_props = []
        for prop_name in _prop_features:
            if prop_name.startswith(_column_name):
                _matched_props.append(prop_name)

        assert _matched_props
        return _matched_props

    for example in examples:
        e_id = example['id']
        column_annotation = rel_pred_results[e_id]['columns']

        for column_name, annotation in column_annotation.items():
            matched_props = _get_matched_props(column_name, example['prop_features'])
            for prop in matched_props:
                is_triggered = annotation['prediction']
                example['prop_features'][prop] = [example['prop_features'][prop][0], is_triggered * 1]

    with open(example_file, 'w') as f:
        for example in examples:
            json_str = json.dumps(example)
            f.write(json_str + '\n')


if __name__ == '__main__':
    # example_file = '/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess_revised/test_split.jsonl'
    # examples = dump_data(example_file,
    #                      '/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess_revised/tables.jsonl')
    #
    # with open(os.path.join(os.path.dirname(example_file), 'test_split') + '.rel_prediction.jsonl', 'w') as f:
    #     for example in examples:
    #         json_str = json.dumps(example)
    #         f.write(json_str + '\n')

    for file_name in glob.glob('/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess_revised/test_split.jsonl'):
        print(file_name)
        load_relation_prediction_results_to_examples(file_name,
                                                     '/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess_revised/test_split.rel_prediction.jsonl.prediction')
