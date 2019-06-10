import argparse
import json
import os
import glob
from pathlib import Path


def load_jsonl(file_path):
    file_path = Path(file_path)
    data = []
    for line in open(file_path):
        entry = json.loads(line)
        data.append(entry)

    return data


def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for example in data:
            json_str = json.dumps(example)
            f.write(json_str + '\n')


def dump_wtq_data_to_bert_input(example_file, table_file, valid_example_ids=None, triggered_columns=None):
    examples = load_jsonl(example_file)
    tables = load_jsonl(table_file)
    tables = {x['name']: x for x in tables}

    triggered_columns = triggered_columns or dict()
    valid_example_ids = valid_example_ids or {e['id'] for e in examples}

    output_examples = []
    for example in examples:
        e_id = example['id']
        if e_id not in valid_example_ids:
            continue

        question = example['question']
        table_id = example['context']
        table = tables[table_id]

        columns = []
        for column in table['props']:
            is_target = e_id in triggered_columns and column in triggered_columns[e_id]

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
                            'is_target': is_target}
            columns.append(column_entry)

        output_examples.append({
            'guid': example['id'],
            'question': question,
            'columns': columns
        })

    return output_examples


def load_relation_prediction_results_to_examples(example_file, relation_prediction_result_file):
    examples = load_jsonl(example_file)
    rel_pred_results = json.load(relation_prediction_result_file.open())

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

    with example_file.open('w') as f:
        for example in examples:
            json_str = json.dumps(example)
            f.write(json_str + '\n')


def main():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    input_parser = sub_parsers.add_parser('generate_input')
    input_parser.set_defaults(which='input')
    input_parser.add_argument('--train-and-dev-examples-file', type=str, required=True)
    input_parser.add_argument('--train-and-dev-tables-file', type=str, required=True)
    input_parser.add_argument('--test-examples-file', type=str, required=True)
    input_parser.add_argument('--test-tables-file', type=str, required=True)
    input_parser.add_argument('--work-dir', type=str, required=True)

    write_parser = sub_parsers.add_parser('write')
    write_parser.set_defaults(which='write')
    write_parser.add_argument('--train-shard-path', type=str, required=True)
    write_parser.add_argument('--test-examples-file', type=str, required=True)
    write_parser.add_argument('--work-dir', type=str, required=True)

    sub_parser = sub_parsers.add_parser('predict_and_generate_parsing_data')
    sub_parser.set_defaults(which='predict_and_generate_parsing_data')
    sub_parser.add_argument('--model-path', type=Path, required=True)

    sub_parser.add_argument('--train-pred-file', type=Path, required=True)
    sub_parser.add_argument('--test-pred-file', type=Path, required=True)

    sub_parser.add_argument('--sp-train-file', type=Path, required=True)
    sub_parser.add_argument('--sp-test-file', type=Path, required=True)

    args = parser.parse_args()

    if args.which == 'input':
        dump_wtq_dataset_for_relation_prediction(args)
    elif args.which == 'write':
        load_wtq_relation_prediction_results(args)
    elif args.which == 'predict_and_generate_parsing_data':
        predict_relations_and_dump_results(args)


def dump_wtq_dataset_for_relation_prediction(args):
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    train_and_dev_examples = dump_wtq_data_to_bert_input(os.path.expanduser(args.train_and_dev_examples_file), os.path.expanduser(args.train_and_dev_tables_file), triggerred_columns=None)
    write_jsonl(train_and_dev_examples, os.path.join(os.path.expanduser(args.work_dir), 'wtq.train_dev.rel_prediction.jsonl'))

    test_examples = dump_wtq_data_to_bert_input(os.path.expanduser(args.test_examples_file), os.path.expanduser(args.test_tables_file))
    write_jsonl(test_examples, os.path.join(os.path.expanduser(args.work_dir), 'wtq.test.rel_prediction.jsonl'))


def predict_relations_and_dump_results(args):
    model_path = args.model_path
    model_dir = model_path.parent
    suffix = model_dir.name

    os.system(f"""
    python relation_predictor.py test \
        {model_path} \
        {args.train_pred_file.expanduser()}
    """)

    os.system(f"""
        python relation_predictor.py test \
            {model_path} \
            {args.test_pred_file.expanduser()}
        """)

    train_shard_path = args.sp_train_file.expanduser()
    tgt_train_folder = str(train_shard_path) + '_' + suffix
    print(f'writing to {tgt_train_folder}')
    os.system(f"cp -r {train_shard_path} {tgt_train_folder}")

    for examples_file in Path(tgt_train_folder).glob('*.jsonl'):
        print(examples_file)
        load_relation_prediction_results_to_examples(examples_file,
                                                     model_dir / 'wtq.train_dev.rel_prediction.jsonl.prediction')

    test_examples_file = args.sp_test_file.expanduser()
    tgt_test_file = Path(str(test_examples_file) + '_' + suffix)
    print(f'writing to {tgt_test_file}')
    os.system(f"cp {test_examples_file} {tgt_test_file}")

    load_relation_prediction_results_to_examples(tgt_test_file,
                                                 model_dir / 'wtq.test.rel_prediction.jsonl.prediction')


def load_wtq_relation_prediction_results(args):
    for examples_file in glob.glob(os.path.expanduser(args.train_shard_path) + '/*.jsonl'):
        print(examples_file)
        load_relation_prediction_results_to_examples(examples_file, os.path.join(os.path.expanduser(args.work_dir), 'wtq.train_dev.rel_prediction.jsonl.prediction'))

    load_relation_prediction_results_to_examples(os.path.expanduser(args.test_examples_file), os.path.join(os.path.expanduser(args.work_dir), 'wtq.test.rel_prediction.jsonl.prediction'))


if __name__ == '__main__':
    main()
