from typing import List, Callable
import inspect
import logging
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from procnet.conf.DocEE_conf import DocEEConfig
from procnet.data_preparer.basic_preparer import BasicPreparer
from procnet.metric.DocEE_metric import DocEEMetric
from procnet.model.basic_model import BasicModel
from procnet.optimizer.basic_optimizer import BasicOptimizer
from procnet.trainer.basic_trainer import BasicTrainer


class DocEEBasicSeqLabelingTrainer(BasicTrainer):
    def __init__(self,
                 config: DocEEConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 test_loader: DataLoader,
                 ):
        super().__init__(config, model, optimizer, preparer, train_loader, dev_loader, test_loader)
        self.result_folder_path = self.result_folder_init(config.model_save_name)
        self.model_save_folder_path = self.checkpoint_folder_init(config.model_save_name)
        self.config = config
        self.preparer = preparer

    def train_batch_template(self,
                             model_run_fn: Callable,
                             dataloader: DataLoader,
                             epoch=-1,
                             ):
        self.model.train()
        start_time = time.time()
        batch_step = 0
        epoch_loss = None
        error_num = 0
        with tqdm(dataloader, unit="b", position=0, leave=True, disable=True) as tqdm_epoch:
            for batch in tqdm_epoch:
                batch_step += 1
                use_mix_bio = False if epoch <= 2 else True
                loss, res = model_run_fn(self.model, batch, run_eval=False, use_mix_bio=use_mix_bio)
                loss.backward()
                self.optimizer.gradient_update()
                epoch_loss = loss.item() if epoch_loss is None else 0.98 * epoch_loss + 0.02 * loss.item()
                for r in res:
                    if 'error_report' in r and r['error_report'] != '':
                        error_num += 1
        used_time = (time.time() - start_time) / 60
        logging.info('Train Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}, Error Report Num = {}'.format(epoch, used_time, epoch_loss, error_num))

    def eval_batch_template(self,
                            model_run_fn: Callable,
                            score_fn: Callable,
                            dataloader: DataLoader,
                            run_eval=True,
                            epoch=-1,
                            ):
        self.model.eval()
        epoch_loss = 0
        start_time = time.time()
        raw_results: List[dict] = []
        for batch in tqdm(dataloader, unit="b", position=0, leave=True, disable=True):
            loss, res = model_run_fn(self.model, batch, run_eval=run_eval, use_mix_bio=False)
            epoch_loss += loss.item()
            raw_results += res
        error_reports = set([x['error_report'] for x in raw_results if x['error_report'] != ''])
        if len(error_reports) > 0:
            logging.warning('Eval error: ' + str(error_reports))
        epoch_loss = epoch_loss / len(dataloader)
        score_to_print, score_result = score_fn(raw_results)
        used_time = (time.time() - start_time) / 60
        error_num = sum([1 if r['error_report'] != '' else 0 for r in raw_results])
        logging.info('Eval Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}, Error Report Num = {}, \nScore = {}'.format(epoch, used_time, epoch_loss, error_num, score_to_print))
        return score_result, raw_results

    def train_template(self,
                       model_run_fn: Callable,
                       score_fn: Callable,
                       train_loader: DataLoader = None,
                       dev_loader: DataLoader = None,
                       test_loader: DataLoader = None,
                       ):
        train_loader = self.train_loader if train_loader is None else train_loader
        dev_loader = self.dev_loader if dev_loader is None else dev_loader
        test_loader = self.test_loader if test_loader is None else test_loader
        for epoch in range(1, self.config.max_epochs + 1):
            epoch_formatted = self.epoch_format(epoch, 3)
            self.train_batch_template(model_run_fn, dataloader=train_loader, epoch=epoch)
            # save model checkpoint
            model_save_path = self.model_save_folder_path / (self.config.model_save_name + '_' + epoch_formatted + '.pth')
            self.optimizer.save_model(model_save_path)
            logging.info('Eval Epoch = {}, dev:'.format(epoch))
            dev_score_results, dev_raw_results = self.eval_batch_template(model_run_fn, score_fn=score_fn, dataloader=dev_loader, epoch=epoch)
            logging.info('Eval Epoch = {}, test:'.format(epoch))
            test_score_results, test_raw_results = self.eval_batch_template(model_run_fn, score_fn=score_fn, dataloader=test_loader, epoch=epoch)
            final_score_results = {'dev': dev_score_results,
                                   'test': test_score_results,
                                   "epoch": epoch,
                                   }
            score_results_file_name = self.config.model_save_name + '_' + epoch_formatted + '.json'
            self.write_json_file(self.result_folder_path / score_results_file_name, final_score_results)


class DocEETrainer(DocEEBasicSeqLabelingTrainer):
    def __init__(self,
                 config: DocEEConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 preparer: BasicPreparer,
                 metric: DocEEMetric,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 test_loader: DataLoader,
                 ):
        super().__init__(config, model, optimizer, preparer, train_loader, dev_loader, test_loader)
        self.metric = metric
        self.score_fn = metric.the_score_fn
        self.return_procnet_entity_nodes = getattr(config, 'return_procnet_entity_nodes', False)
        self.use_procnet_entity_nodes = getattr(config, 'use_procnet_entity_nodes', False)
        self.model_accepts_procnet_entity_nodes = 'procnet_entity_nodes' in inspect.signature(model.forward).parameters
        self._warned_procnet_entity_nodes_ignored = False

    def _unpack_batch(self, batch: list):
        if len(batch) >= 6:
            doc_id, input_ids, input_att_masks, bio_ids, events_labels, procnet_entity_nodes = (b for b in batch[:6])
        else:
            doc_id, input_ids, input_att_masks, bio_ids, events_labels = (b for b in batch)
            procnet_entity_nodes = None
        return doc_id, input_ids, input_att_masks, bio_ids, events_labels, procnet_entity_nodes

    def model_fn(self, model: BasicModel, batch: list, run_eval: bool, use_mix_bio: bool):
        doc_id, input_ids, input_att_masks, bio_ids, events_labels, procnet_entity_nodes = self._unpack_batch(batch)
        doc_id_show = doc_id
        if isinstance(doc_id_show, (list, tuple)) and len(doc_id_show) == 1:
            doc_id_show = doc_id_show[0]

        batch_len = len(batch) if isinstance(batch, (list, tuple)) else None

        if procnet_entity_nodes is None:
            logging.debug(
                "[PROCNET_DEBUG][trainer_in] doc_id=%s run_eval=%s batch_len=%s has_procnet_entity_nodes=False",
                doc_id_show, run_eval, batch_len
            )
        else:
            fragment_node_counts = []
            first_non_empty_node = None
            for one_fragment_nodes in procnet_entity_nodes:
                cnt = len(one_fragment_nodes) if one_fragment_nodes is not None else -1
                fragment_node_counts.append(cnt)
                if first_non_empty_node is None and one_fragment_nodes:
                    first_non_empty_node = one_fragment_nodes[0]

            non_empty_fragment_num = sum(1 for x in procnet_entity_nodes if x)

            logging.debug(
                "[PROCNET_DEBUG][trainer_in] doc_id=%s run_eval=%s batch_len=%s has_procnet_entity_nodes=True "
                "fragment_num=%s non_empty_fragment_num=%s fragment_node_counts=%s first_node_keys=%s",
                doc_id_show,
                run_eval,
                batch_len,
                len(procnet_entity_nodes),
                non_empty_fragment_num,
                fragment_node_counts,
                sorted(first_non_empty_node.keys()) if isinstance(first_non_empty_node, dict) else None,
            )

        input_ids = input_ids.to(self.device) if isinstance(input_ids, torch.Tensor) else [x.to(self.device) for x in input_ids]
        input_att_masks = input_att_masks.to(self.device) if isinstance(input_att_masks, torch.Tensor) else None

        model_kwargs = {
            'inputs_ids': input_ids,
            'inputs_att_masks': input_att_masks,
            'events_labels': events_labels,
        }
        if not run_eval:
            bio_ids_run = bio_ids.to(self.device) if isinstance(bio_ids, torch.Tensor) else [x.to(self.device) for x in bio_ids]
            model_kwargs['bios_ids'] = bio_ids_run
            model_kwargs['use_mix_bio'] = use_mix_bio

        should_try_procnet_nodes = procnet_entity_nodes is not None and (
            self.return_procnet_entity_nodes or self.use_procnet_entity_nodes
        )
        if should_try_procnet_nodes:
            if self.model_accepts_procnet_entity_nodes:
                model_kwargs['procnet_entity_nodes'] = procnet_entity_nodes
            elif not self._warned_procnet_entity_nodes_ignored:
                logging.warning(
                    'Batch already carries procnet_entity_nodes, but model.forward has no procnet_entity_nodes kwarg yet. '
                    'Trainer will ignore them for now.'
                )
                self._warned_procnet_entity_nodes_ignored = True

        model_res = model(**model_kwargs)
        loss, result = model_res

        node_num_from_batch = [len(x) for x in procnet_entity_nodes] if procnet_entity_nodes is not None else None
        used_procnet_entity_nodes = result.get('used_procnet_entity_nodes', None)
        error_report = result.get('error_report', None)

        logging.debug(
            "[PROCNET_DEBUG][trainer_out] doc_id=%s run_eval=%s used_procnet_entity_nodes=%s "
            "error_report=%s loss=%.6f node_num_from_batch=%s",
            doc_id_show,
            run_eval,
            used_procnet_entity_nodes,
            error_report,
            float(loss.detach().item()) if hasattr(loss, "detach") else float(loss),
            node_num_from_batch,
        )

        if used_procnet_entity_nodes or (error_report not in [None, '']):
            logging.info(
                "[PROCNET_HIT] doc_id=%s run_eval=%s used_procnet_entity_nodes=%s "
                "error_report=%s loss=%.6f node_num_from_batch=%s",
                doc_id_show,
                run_eval,
                used_procnet_entity_nodes,
                error_report,
                float(loss.detach().item()) if hasattr(loss, "detach") else float(loss),
                node_num_from_batch,
            )

        if isinstance(bio_ids, torch.Tensor):
            BIO_ans = bio_ids.view(-1).detach().cpu().numpy().tolist()
        else:
            BIO_ans = torch.cat(bio_ids, dim=0).view(-1).detach().cpu().numpy().tolist()
        assert len(BIO_ans) == len(result['BIO_pred'])
        events_label = events_labels
        other_record = {
               'doc_id': doc_id,
               'BIO_ans': BIO_ans,
               'event_ans': events_label,
               }
        if procnet_entity_nodes is not None:
            other_record.update({
                'has_procnet_entity_nodes': True,
                'procnet_entity_node_num': [len(x) for x in procnet_entity_nodes],
            })
        result.update(other_record)
        return loss, [result]

    def train(self):
        self.train_template(model_run_fn=self.model_fn,
                            score_fn=self.score_fn,
                            )

    def eval(self,
             test_loader: DataLoader = None,
             true_bio: bool = False,
             ):
        test_loader = self.test_loader if test_loader is None else test_loader
        if true_bio:
            score_result, raw_results = self.eval_batch_template(model_run_fn=self.model_fn,
                                                                 score_fn=self.score_fn,
                                                                 dataloader=test_loader,
                                                                 epoch='Test',
                                                                 run_eval=False,
                                                                 )
        else:
            score_result, raw_results = self.eval_batch_template(model_run_fn=self.model_fn,
                                                                 score_fn=self.score_fn,
                                                                 dataloader=test_loader,
                                                                 epoch='Test',
                                                                 )
        return score_result, raw_results
