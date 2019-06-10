import argparse
from pathlib import Path
import os

from table.bert.dump_wtq_data import predict_relations_and_dump_results, dump_wtq_data_to_bert_input, write_jsonl


def train(args):
    pwd = os.getcwd()
    os.chdir('table/bert')
    new_train_folder, new_test_file = predict_relations_and_dump_results(args.bert_model_path,
                                                                         args.train_pred_file,
                                                                         args.test_pred_file,
                                                                         args.sp_train_file,
                                                                         args.sp_test_file)
    os.chdir(pwd)

    nsm_work_dir = args.work_dir
    cmd = f"""
            OMP_NUM_THREADS=1 \
            python -m table.experiments \
                train \
                --cuda \
                --work-dir={nsm_work_dir} \
                --extra-config='{{"train_shard_dir": "{new_train_folder}", "dev_file": "{new_train_folder / 'dev_split.jsonl'}"}}' \
                --config=table/config.rel_annot.json 2>{nsm_work_dir / 'err.log'}
        """
    os.system(cmd)

    cmd = f"""
            OMP_NUM_THREADS=1 \
            python -m table.experiments \
                test \
                --cuda \
                --model={nsm_work_dir / 'model.best.bin'} \
                --test-file={new_test_file} \
                --save-decode-to={nsm_work_dir / 'decode.test.json'} 2>{nsm_work_dir / 'err.test.log'}
        """
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--bert-model-path', type=Path, required=True)

    parser.add_argument('--train-pred-file', type=Path, required=True)
    parser.add_argument('--test-pred-file', type=Path, required=True)

    parser.add_argument('--sp-train-file', type=Path, required=True)
    parser.add_argument('--sp-test-file', type=Path, required=True)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
