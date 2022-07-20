import warnings

warnings.filterwarnings(action="ignore")

import hydra
import wandb
from datasets import load_metric
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from dataloader import load


@hydra.main(config_name="config.yml")
def main(cfg):
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.PATH.model_name)

    # dataloader
    train_dataset, eval_dataset = load(tokenizer=tokenizer, **cfg.DATASETS)

    # model
    model = AutoModelForTokenClassification.from_pretrained(
        cfg.PATH.model_name, num_labels=13
    )

    wandb.init(
        project=cfg.ETC.wandb_project,
        entity=cfg.ETC.wandb_entity,
        name=cfg.ETC.wandb_name,
    )

    # trainer
    args = TrainingArguments(
        do_train=True,
        do_eval=eval_dataset is not None,
        logging_dir=cfg.PATH.logging_dir,
        output_dir=cfg.PATH.checkpoint_dir,
        **cfg.TRAININGARGS,
    )

    metric = load_metric(cfg.METRICS.metric_name)

    def compute_metrics(pred):
        logits, labels = pred
        for x, y in zip(logits.argmax(-1), labels):
            metric.add_batch(predictions=x, references=y)
        return metric.compute(average=cfg.METRICS.average)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    # train
    trainer.train()

    model.save_pretrained(cfg.PATH.output_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
