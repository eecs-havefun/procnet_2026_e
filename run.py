import os
import importlib
import torch
import logging
import argparse

from procnet.utils.util_string import UtilString
from procnet.data_processor.DocEE_processor import DocEEProcessor
from procnet.data_preparer.DocEE_preparer import DocEEPreparer
from procnet.model.DocEE_proxy_node_model import DocEEProxyNodeModel
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.conf.DocEE_conf import DocEEConfig

importlib.reload(logging)
logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.INFO,
    datefmt='%I:%M:%S'
)


def str_to_bool(x):
    if isinstance(x, bool):
        return x
    return UtilString.str_to_bool(x)


def normalize_optional_path(path_value):
    if path_value is None:
        return None
    v = str(path_value).strip()
    if v == "" or v.lower() in {"none", "null"}:
        return None
    return os.path.abspath(v)


def parse_args(in_args=None):
    repo_root = os.path.dirname(os.path.abspath(__file__))

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--run_save_name",
        type=str,
        required=True,
        help="The save name of this run"
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="gradient_accumulation_steps in current ProcNet code"
    )
    arg_parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="Training epochs"
    )
    arg_parser.add_argument(
        "--read_pseudo",
        type=str,
        default="false",
        help="If read pseudo data"
    )

    arg_parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(repo_root, "Data"),
        help="Path to dataset directory"
    )
    arg_parser.add_argument(
        "--typed_entities_dir",
        type=str,
        default=os.path.join(repo_root, "tmp_sidecar"),
        help="Path to typed-entity sidecar directory, e.g. ./tmp_sidecar or ./sidecar"
    )

    arg_parser.add_argument(
        "--use_procnet_entity_nodes",
        type=str,
        default="true",
        help="Whether model uses procnet sidecar entity nodes"
    )
    arg_parser.add_argument(
        "--return_procnet_entity_nodes",
        type=str,
        default=None,
        help="Whether processor/preparer returns procnet sidecar nodes; defaults to use_procnet_entity_nodes when omitted"
    )
    arg_parser.add_argument(
        "--use_procnet_pred_entities",
        type=str,
        default="true",
        help="Whether processor loads procnet typed-entity sidecar"
    )

    arg_parser.add_argument(
        "--proxy_slot_num",
        type=int,
        default=16,
        help="Proxy slot num"
    )
    arg_parser.add_argument(
        "--node_size",
        type=int,
        default=512,
        help="Node hidden size"
    )
    arg_parser.add_argument(
        "--max_len",
        type=int,
        default=510,
        help="Max token length"
    )
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="Backbone model name or local model path"
    )
    arg_parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Training device"
    )
    arg_parser.add_argument(
        "--data_loader_shuffle",
        type=str,
        default="true",
        help="Whether dataloader shuffles training data"
    )

    args = arg_parser.parse_args(args=in_args)

    args.read_pseudo = str_to_bool(args.read_pseudo)
    args.use_procnet_entity_nodes = str_to_bool(args.use_procnet_entity_nodes)
    args.use_procnet_pred_entities = str_to_bool(args.use_procnet_pred_entities)
    args.data_loader_shuffle = str_to_bool(args.data_loader_shuffle)

    if args.return_procnet_entity_nodes is None:
        args.return_procnet_entity_nodes = args.use_procnet_entity_nodes
    else:
        args.return_procnet_entity_nodes = str_to_bool(args.return_procnet_entity_nodes)

    args.dataset_dir = os.path.abspath(args.dataset_dir)
    args.typed_entities_dir = normalize_optional_path(args.typed_entities_dir)

    return args


def get_config(args) -> DocEEConfig:
    config = DocEEConfig()

    config.return_procnet_entity_nodes = args.return_procnet_entity_nodes
    config.use_procnet_entity_nodes = args.use_procnet_entity_nodes
    config.use_procnet_pred_entities = args.use_procnet_pred_entities

    config.proxy_slot_num = args.proxy_slot_num
    config.node_size = args.node_size
    config.max_len = args.max_len

    config.model_save_name = args.run_save_name
    config.gradient_accumulation_steps = args.batch_size
    config.max_epochs = args.epoch
    config.data_loader_shuffle = args.data_loader_shuffle
    config.model_name = args.model_name

    if args.device == "cuda" and torch.cuda.is_available():
        config.device = torch.device("cuda")
    else:
        config.device = torch.device("cpu")

    return config


def validate_paths(args):
    if not os.path.isdir(args.dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {args.dataset_dir}")

    if args.use_procnet_pred_entities:
        if args.typed_entities_dir is None:
            raise ValueError(
                "use_procnet_pred_entities=True but typed_entities_dir is None"
            )
        if not os.path.isdir(args.typed_entities_dir):
            raise FileNotFoundError(
                f"typed_entities_dir not found: {args.typed_entities_dir}"
            )


def run(args):
    validate_paths(args)
    config = get_config(args)

    logging.info("========== Run Config ==========")
    logging.info("save_name = %s", config.model_save_name)
    logging.info("dataset_dir = %s", args.dataset_dir)
    logging.info("typed_entities_dir = %s", args.typed_entities_dir)
    logging.info("read_pseudo = %s", args.read_pseudo)
    logging.info("return_procnet_entity_nodes = %s", config.return_procnet_entity_nodes)
    logging.info("use_procnet_entity_nodes = %s", config.use_procnet_entity_nodes)
    logging.info("use_procnet_pred_entities = %s", config.use_procnet_pred_entities)
    logging.info("batch_size(grad_accum_steps) = %s", config.gradient_accumulation_steps)
    logging.info("max_epochs = %s", config.max_epochs)
    logging.info("proxy_slot_num = %s", config.proxy_slot_num)
    logging.info("node_size = %s", config.node_size)
    logging.info("max_len = %s", config.max_len)
    logging.info("model_name = %s", config.model_name)
    logging.info("device = %s", config.device)
    logging.info("================================")

    dee_pro = DocEEProcessor(
        read_pseudo_dataset=args.read_pseudo,
        use_procnet_pred_entities=config.use_procnet_pred_entities,
        dataset_dir=args.dataset_dir,
        typed_entities_dir=args.typed_entities_dir,
    )

    dee_pre = DocEEPreparer(config=config, processor=dee_pro)
    pre_data = dee_pre.get_loader_for_flattened_fragment_before_event()
    train_dataset, dev_dataset, test_dataset, train_loader, dev_loader, test_loader = pre_data

    metric = DocEEMetric(preparer=dee_pre)
    model = DocEEProxyNodeModel(config=config, preparer=dee_pre)
    model.to(config.device)

    optimizer = BasicOptimizer(config=config, model=model)

    trainer = DocEETrainer(
        config=config,
        model=model,
        optimizer=optimizer,
        preparer=dee_pre,
        metric=metric,
        train_loader=train_loader,
        dev_loader=dev_loader,
        test_loader=test_loader,
    )
    trainer.train()


if __name__ == '__main__':
    arg = parse_args()
    run(arg)
