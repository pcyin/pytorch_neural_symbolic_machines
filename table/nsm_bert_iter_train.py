import argparse
from pathlib import Path

from table.bert.dump_wtq_data import predict_relations_and_dump_results, dump_wtq_data_to_bert_input, write_jsonl
from table.experiments import *
import os
import subprocess


def train(args):
    wtq_base_dir = args.wtq_base_dir
    envs = load_environments([wtq_base_dir / 'train_examples.jsonl'],
                             table_file=str(wtq_base_dir / 'tables.jsonl'),
                             vocab_file=str((wtq_base_dir / '../../raw_input/wikitable_glove_vocab.json').resolve()),
                             en_vocab_file=str(wtq_base_dir / 'en_vocab_min_count_5.json'),
                             embedding_file=str((wtq_base_dir / '../../raw_input/wikitable_glove_embedding_mat.npy').resolve()))
    env_dict = {env.name: env for env in envs}
    pwd = os.getcwd()

    prev_bert_dir = args.init_bert_dir
    for epoch in range(5):
        os.chdir('table/bert')
        # use TableBERT to make predictions -> NSM training/test files
        new_sp_train_folder, new_sp_test_file = predict_relations_and_dump_results(prev_bert_dir / 'pytorch_model.bin',
                                                                                   args.bert_init_pred_file_train,
                                                                                   args.bert_init_pred_file_test,
                                                                                   wtq_base_dir / 'data_split_1/',
                                                                                   wtq_base_dir / 'test_split.jsonl',
                                                                                   suffix=args.work_dir.name + f'_epoch{epoch}')
        os.chdir(pwd)

        # train NSM using new BERT predictions
        nsm_work_dir = Path(str(args.work_dir) + f'_epoch{epoch}')
        nsm_work_dir.mkdir(exist_ok=True, parents=True)

        if not (nsm_work_dir / 'model.best.bin').exists():
            os.system(f"""
                OMP_NUM_THREADS=1 \
                python -m table.experiments \
                    train \
                    --cuda \
                    --work-dir={nsm_work_dir} \
                    --extra-config='{{"train_shard_dir": "{new_sp_train_folder}", "dev_file": "{new_sp_train_folder / 'dev_split.jsonl'}"}}' \
                    --config=table/config.rel_annot.json 2>{nsm_work_dir / 'err.log'}
            """)

        # dump NSM predictions for BERT training
        for (tag, test_file, decode_save_file) in [('train', new_sp_train_folder / 'train_split.jsonl', nsm_work_dir / 'decode.train.json'),
                                                   ('test', new_sp_test_file, nsm_work_dir / 'decode.test.json')]:
            if not decode_save_file.exists():
                os.system(f"""
                            OMP_NUM_THREADS=1 \
                            python -m table.experiments \
                                test \
                                --cuda \
                                --model={nsm_work_dir / 'model.best.bin'} \
                                --test-file={test_file} \
                                --save-decode-to={decode_save_file} 2>{nsm_work_dir / ('err.' + tag + '.log')}
                """)

        # NSM predictions to BERT training files
        triggered_columns = {}
        train_decode_results = json.load((nsm_work_dir / 'decode.train.json').open())   # train split only, no dev
        for env_id in env_dict:
            if env_id not in train_decode_results:
                continue

            env = env_dict[env_id]
            hyps = train_decode_results[env_id]
            if hyps and hyps[0]['is_correct']:
                program = hyps[0]['program']

                program = to_human_readable_program(program, env)
                _triggered_columns = set()
                for token in program:
                    token = str(token)
                    if token.startswith('r.'):
                        _triggered_columns.add(token)

                if _triggered_columns:
                    triggered_columns[env_id] = list(_triggered_columns)

        bert_examples = dump_wtq_data_to_bert_input(
            new_sp_train_folder / 'train_split.jsonl',
            wtq_base_dir / 'tables.jsonl',
            valid_example_ids=set(triggered_columns),
            triggered_columns=triggered_columns)

        # Start BERT training for one epoch
        os.chdir('table/bert')

        bert_dir_for_cur_epoch = prev_bert_dir.parent / (prev_bert_dir.name + f'_finetune_epoch{epoch}' )
        bert_dir_for_cur_epoch.mkdir(exist_ok=True)

        bert_train_file = bert_dir_for_cur_epoch / f'wtq_train_examples.epoch{epoch}.jsonl'
        write_jsonl(bert_examples, bert_train_file)

        os.system(f"""
                   python relation_predictor.py train \
                        --extra-config='{{"pretrained_model_path": "{prev_bert_dir / 'pytorch_model.bin'}", "data": {{ "train_file": "{bert_train_file}", "dev_file": "{args.bert_init_pred_file_test}" }}, "output_dropout_prob": 0.1}}' \
                        --seed=0 \
                        --work-dir={bert_dir_for_cur_epoch} \
                        config.local.json
                """)
        os.chdir(pwd)

        prev_bert_dir = bert_dir_for_cur_epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, required=False)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--init-bert-dir', type=Path, required=True)
    parser.add_argument('--wtq-base-dir', type=Path, required=True)

    parser.add_argument('--bert-init-pred-file-train', type=Path, required=True)
    parser.add_argument('--bert-init-pred-file-test', type=Path, required=True)

    args = parser.parse_args()

    train(args)
