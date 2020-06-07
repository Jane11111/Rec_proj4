# -*- coding: utf-8 -*-
# @Time    : 2020-05-15 22:52
# @Author  : zxl
# @FileName: test.py

import configargparse
# CONFIG_FILE='model.yml'
def preclean_opt(parse):
    group = parse.add_argument_group("Preclean")
    group.add("--device", "-device", type=str)
    group.add("--word2vec_path", "-word2vec_path", type=str)
    group.add("--max_voc_len", "-max_voc_len", type=int)

    group.add("--vocab_size", "-vocab_size", type=int)
    group.add("--embedding_size", "-embedding_size", type=int)
    group.add("--k_conv_emb", "-k_conv_emb", type=int)
    group.add("--window_size", "-window_size", type=int)
    group.add("--k_id_emb", "-k_id_emb", type=int)
    group.add("--k_att", "-k_att", type=int)
    group.add("--k_lfm", "-k_lfm", type=int)
    group.add("--batch_size", "-batch_size", type=int)
    group.add("--lr", "-lr", type=float)
    group.add("--save_path", "-save_path", type=str)
    group.add("--weight_decay", "-weight_decay", type=float)
    group.add("--epoch", "-epoch", type=int)

    return parse


def get_config(path):
    parse = configargparse.ArgumentParser(default_config_files=[path],
                                              config_file_parser_class=configargparse.YAMLConfigFileParser)
    parse = preclean_opt(parse)
    config,unknown = parse.parse_known_args()
    # print(type(config.vocab_size))
    return config