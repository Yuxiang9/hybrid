from functools import partial
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.deep_impact.models import DeepImpact, DeepPairwiseImpact, DeepImpactCrossEncoder, HybridDeepImpact, MetaEmbedDeepImpact
from src.deep_impact.training import Trainer, PairwiseTrainer, CrossEncoderTrainer, DistilTrainer, \
    InBatchNegativesTrainer, HybridDistilTrainer, MetaEmbedKDTrainer
from src.deep_impact.training.distil_trainer import DistilMarginMSE, DistilKLLoss
from src.deep_impact.training.hybrid_distil_trainer import hybrid_distil_collate_fn
from src.deep_impact.training.meta_embed_kd_trainer import meta_embed_kd_collate_fn
from src.utils.datasets import MSMarcoTriples, DistillationScores, DistillationScoresToTriples
from src.deep_impact.evaluation.nano_beir_evaluator import NanoBEIREvaluator
from src.deep_impact.training.dense_trainer import DenseTrainer, dense_joint_collate_fn, DenseBiEncoder


def collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks = [], []
    for query, positive_document, negative_document in batch:
        encoded_token, mask = model_cls.process_query_and_document(query, positive_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)

        encoded_token, mask = model_cls.process_query_and_document(query, negative_document, max_length=max_length)
        encoded_list.append(encoded_token)
        masks.append(mask)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
    }


def cross_encoder_collate_fn(batch):
    encoded_list = []
    for query, positive_document, negative_document in batch:
        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(positive_document, query)
        encoded_list.append(encoded_token)

        encoded_token = DeepImpactCrossEncoder.process_cross_encoder_document_and_query(negative_document, query)
        encoded_list.append(encoded_token)

    return {'encoded_list': encoded_list}


def distil_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    encoded_list, masks, scores = [], [], []
    for query, pid_score_list in batch:
        for passage, score in pid_score_list:
            encoded_token, mask = model_cls.process_query_and_document(query, passage, max_length=max_length)
            encoded_list.append(encoded_token)
            masks.append(mask)
            scores.append(score)

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0).unsqueeze(-1),
        'scores': torch.tensor(scores, dtype=torch.float),
    }


def in_batch_negatives_collate_fn(batch, model_cls=DeepImpact, max_length=None):
    queries, positive_documents, negative_documents = zip(*batch)
    queries_terms = [model_cls.process_query(query) for query in queries]
    negatives = [model_cls.process_document(document) for document in negative_documents]

    encoded_list, masks = [], []
    for i, (query_terms, positive_document) in enumerate(zip(queries_terms, positive_documents)):
        encoded_token, term_to_token_index = model_cls.process_document(positive_document)

        encoded_list.append(encoded_token)
        masks.append(model_cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length))

        encoded_list.append(negatives[i][0])
        for _, term_to_token_index in negatives:
            masks.append(model_cls.get_query_document_token_mask(query_terms, term_to_token_index, max_length))

    return {
        'encoded_list': encoded_list,
        'masks': torch.stack(masks, dim=0),
    }


def run(
        dataset_path: Union[str, Path],
        queries_path: Union[str, Path],
        collection_path: Union[str, Path],
        checkpoint_dir: Union[str, Path],
        max_length: int,
        seed: int,
        batch_size: int,
        lr: float,
        save_every: int,
        save_best: bool,
        gradient_accumulation_steps: int,
        pairwise: bool = False,
        cross_encoder: bool = False,
        distil_mse: bool = False,
        distil_kl: bool = False,
        dense: bool = False,
        in_batch_negatives: bool = False,
        hybrid: bool = False,
        meta_embed: bool = False,
        meta_embed_kd: bool = False,
        expansion_weight: float = 0.3,
        regular_weight: float = 0.7,
        start_with: Union[str, Path] = None,
        qrels_path: Union[str, Path] = None,
        eval_every: int = 500,
        # MetaEmbed-specific parameters
        dense_dim: int = 128,
        num_expansion_tokens: int = 64,
        sparse_loss_weight: float = 1.0,
        dense_loss_weight: float = 1.0,
        combined_loss_weight: float = 1.0,
        loss_temperature: float = 1.0,
        fixed_weights: bool = False,
        # MetaEmbed-KD specific parameters
        lambda_di: float = 1.0,
        lambda_li: float = 1.0,
        lambda_kd: float = 1.0,
        margin: float = 0.2,
        kd_temperature: float = 1.0,
        num_hard_negatives: int = 8,
        top_k_hard: int = 4,
):
    # DeepImpact
    model_cls = DeepImpact
    trainer_cls = Trainer
    collate_function = partial(collate_fn, model_cls=DeepImpact, max_length=max_length)
    dataset_cls = MSMarcoTriples

    # Pairwise
    if pairwise:
        model_cls = DeepPairwiseImpact
        trainer_cls = PairwiseTrainer
        collate_function = partial(collate_fn, model_cls=DeepPairwiseImpact, max_length=max_length)

    # CrossEncoder
    elif cross_encoder:
        model_cls = DeepImpactCrossEncoder
        trainer_cls = CrossEncoderTrainer
        collate_function = cross_encoder_collate_fn

    elif dense:
        model_cls = DeepImpact
        trainer_cls = DenseTrainer
        trainer_cls.loss = DistilKLLoss()
        collate_function = partial(dense_joint_collate_fn, max_length_sparse=max_length)
        dataset_cls = DistillationScores

    # MetaEmbed with Knowledge Distillation (DeeperImpact-style)
    if meta_embed_kd:
        model_cls = MetaEmbedDeepImpact
        trainer_cls = MetaEmbedKDTrainer
        collate_function = partial(
            meta_embed_kd_collate_fn, 
            model_cls=MetaEmbedDeepImpact, 
            max_length=max_length,
            num_hard_negatives=num_hard_negatives,  # Configurable via --num_hard_negatives
            top_k_hard=top_k_hard  # Configurable via --top_k_hard
        )
        # Use DistillationScores (same as DeeperImpact KD) - provides 1 pos + ~50 hard negatives
        # We sample only 8 negatives (4 top + 4 random) to reduce memory usage
        dataset_cls = DistillationScores
    
    
    # Hybrid method with dual scoring and distillation
    elif hybrid:
        model_cls = HybridDeepImpact
        trainer_cls = HybridDistilTrainer
        trainer_cls.loss = DistilKLLoss()
        collate_function = partial(hybrid_distil_collate_fn, model_cls=HybridDeepImpact, max_length=max_length)
        dataset_cls = DistillationScores
    
    # Use distillation loss
    elif distil_mse:
        trainer_cls = DistilTrainer
        trainer_cls.loss = DistilMarginMSE()
        collate_function = partial(distil_collate_fn, max_length=max_length)
        dataset_cls = partial(DistillationScores, qrels_path=qrels_path)
    elif distil_kl:
        trainer_cls = DistilTrainer
        trainer_cls.loss = DistilKLLoss()
        collate_function = partial(distil_collate_fn, max_length=max_length)
        dataset_cls = DistillationScores

    if in_batch_negatives:
        trainer_cls = InBatchNegativesTrainer
        collate_function = partial(in_batch_negatives_collate_fn, max_length=max_length)

    trainer_cls.ddp_setup()

    # Assertions to prevent incompatible training flags
    if meta_embed_kd and meta_embed:
        raise ValueError("Cannot use both --meta_embed and --meta_embed_kd. Choose one.")
    if meta_embed_kd and hybrid:
        raise ValueError("Cannot use both --meta_embed_kd and --hybrid. Choose one.")
    
    # Determine whether to learn weights (default True unless --fixed_weights is specified)
    learn_weights = not fixed_weights if 'fixed_weights' in locals() else True
    
    if start_with:
        if model_cls == MetaEmbedDeepImpact:
            model = model_cls.load(start_with, dense_dim=dense_dim, 
                                 num_expansion_tokens=num_expansion_tokens,
                                 learn_weights=learn_weights)
        else:
            model = model_cls.load(start_with)
    else:
        # For MetaEmbedDeepImpact, we need to pass configuration parameters
        if model_cls == MetaEmbedDeepImpact:
            from transformers import AutoConfig

            print(f"Setting {num_expansion_tokens} expansion tokens...")
            model_cls.set_expansion_tokens(num_expansion_tokens)

            # Re-enable padding/truncation AFTER adding tokens
            # model_cls.tokenizer.enable_truncation(max_length=max_length)
            # model_cls.tokenizer.enable_padding(length=max_length)
            model_cls.tokenizer.model_max_length = max_length

            config = AutoConfig.from_pretrained("Luyu/co-condenser-marco")
            model = model_cls(
                config,
                dense_dim=dense_dim,
                num_expansion_tokens=num_expansion_tokens,
                learn_weights=learn_weights,
            )

            vocab_size = model_cls.tokenizer.vocab_size
            if vocab_size != model.bert.embeddings.word_embeddings.num_embeddings:
                print(f"Resizing token embeddings {model.bert.embeddings.word_embeddings.num_embeddings} -> {vocab_size}")
                model.resize_token_embeddings(vocab_size)

            # Load base DeepImpact for weight initialization
            base_model = DeepImpact.load()

            # Copy BERT weights safely
            base_state = base_model.bert.state_dict()
            model_state = model.bert.state_dict()
            for key, val in base_state.items():
                if key.startswith("embeddings.word_embeddings.weight"):
                    min_vocab = min(val.size(0), model_state[key].size(0))
                    model_state[key][:min_vocab] = val[:min_vocab]
                elif key in model_state and val.size() == model_state[key].size():
                    model_state[key] = val

            model.bert.load_state_dict(model_state)

            # Copy sparse head
            model.impact_score_encoder.load_state_dict(
                base_model.impact_score_encoder.state_dict()
            )

            print("✓ MetaEmbed initialized correctly!")
            
            # Print configuration
            if learn_weights:
                print(f"✓ MetaEmbed initialized with LEARNABLE weights (will adapt during training)")
            else:
                print(f"✓ MetaEmbed initialized with FIXED weights (0.5/0.5, will NOT change during training)")
        # For HybridDeepImpact, we need to pass the weights during initialization
        elif model_cls == HybridDeepImpact:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained('Luyu/co-condenser-marco')
            model = model_cls(config, expansion_weight=expansion_weight, regular_weight=regular_weight)
            # Load pretrained weights from base model
            base_model = DeepImpact.load()
            # Copy BERT weights
            model.bert.load_state_dict(base_model.bert.state_dict())
            # Initialize regular impact encoder from base model
            model.regular_impact_score_encoder.load_state_dict(base_model.impact_score_encoder.state_dict())
        else:
            model = model_cls.load()
    
    # model_cls.tokenizer.enable_truncation(max_length=max_length, strategy='longest_first')
    # model_cls.tokenizer.enable_padding(length=max_length)
    model_cls.tokenizer.model_max_length = max_length

    dataset = dataset_cls(dataset_path, queries_path, collection_path)
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_function,
        sampler=DistributedSampler(dataset),
        drop_last=True,
        num_workers=0,
    )



    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    evaluator = NanoBEIREvaluator(batch_size=64, verbose=False)
    
    # --------------------------------------------------------------
    # Instantiate the appropriate trainer.
    # DenseTrainer expects *two* separate models (sparse & dense) so
    # we branch here accordingly; for all other trainers we keep the
    # original call signature.
    # --------------------------------------------------------------

    if trainer_cls is DenseTrainer:
        dense_model = DenseBiEncoder()
        trainer = trainer_cls(
            sparse_model=model,
            dense_model=dense_model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
        )
    elif trainer_cls is MetaEmbedKDTrainer:
        trainer = trainer_cls(
            model=model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
            lambda_di=lambda_di,
            lambda_li=lambda_li,
            lambda_kd=lambda_kd,
            margin=margin,
            temperature=kd_temperature,
        )
    elif trainer_cls is MetaEmbedTrainer:
        trainer = trainer_cls(
            model=model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
            sparse_weight=sparse_loss_weight,
            dense_weight=dense_loss_weight,
            combined_weight=combined_loss_weight,
            temperature=loss_temperature,
        )
    else:
        trainer = trainer_cls(
            model=model,
            optimizer=optimizer,
            train_data=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            save_every=save_every,
            save_best=save_best,
            seed=seed,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluator=evaluator,
            eval_every=eval_every,
        )
    trainer.train()
    trainer_cls.ddp_cleanup()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Distributed Training of DeepImpact on MS MARCO triples dataset")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Path to the training dataset")
    parser.add_argument("--queries_path", type=Path, required=True, help="Path to the queries dataset")
    parser.add_argument("--collection_path", type=Path, required=True, help="Path to the collection dataset")
    parser.add_argument("--checkpoint_dir", type=Path, required=True, help="Directory to store and load checkpoints")
    parser.add_argument("--max_length", type=int, default=300, help="Max Number of tokens in document")
    parser.add_argument("--seed", type=int, default=42, help="Fix seed")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--save_every", type=int, default=20000, help="Save checkpoint every n steps")
    parser.add_argument("--save_best", action="store_true", help="Save the best model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--pairwise", action="store_true", help="Use pairwise training")
    parser.add_argument("--cross_encoder", action="store_true", help="Use cross encoder model")
    parser.add_argument("--distil_mse", action="store_true", help="Use distillation loss with Mean Squared Error")
    parser.add_argument("--distil_kl", action="store_true", help="Use distillation loss with KL divergence loss")
    parser.add_argument("--dense", action="store_true", help="Use dense training")
    parser.add_argument("--in_batch_negatives", action="store_true", help="Use in-batch negatives")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid method with dual scoring (regular + expansion tokens)")
    parser.add_argument("--meta_embed", action="store_true", help="Use MetaEmbed method with sparse + dense late interaction")
    parser.add_argument("--meta_embed_kd", action="store_true", help="Use MetaEmbed with KD (DeeperImpact-style: Margin + InfoNCE + KD)")
    parser.add_argument("--fixed_weights", action="store_true", help="Use fixed combination weights (0.5/0.5) instead of learning them")
    parser.add_argument("--expansion_weight", type=float, default=0.3, help="Initial weight for expansion token scores (w2)")
    parser.add_argument("--regular_weight", type=float, default=0.7, help="Initial weight for regular term scores (w1)")
    parser.add_argument("--start_with", type=Path, default=None, help="Start training with this checkpoint")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluate every n steps")
    
    # MetaEmbed-specific parameters
    parser.add_argument("--dense_dim", type=int, default=128, help="Dimensionality of dense embeddings for MetaEmbed")
    parser.add_argument("--num_expansion_tokens", type=int, default=64, help="Number of expansion tokens for MetaEmbed")
    parser.add_argument("--sparse_loss_weight", type=float, default=1.0, help="Weight for sparse loss in MetaEmbed")
    parser.add_argument("--dense_loss_weight", type=float, default=1.0, help="Weight for dense loss in MetaEmbed")
    parser.add_argument("--combined_loss_weight", type=float, default=1.0, help="Weight for combined loss in MetaEmbed")
    parser.add_argument("--loss_temperature", type=float, default=1.0, help="Temperature for InfoNCE loss in MetaEmbed")
    
    # MetaEmbed-KD specific parameters (DeeperImpact-style distillation)
    parser.add_argument("--lambda_di", type=float, default=1.0, help="Weight for DeepImpact margin loss in MetaEmbed-KD")
    parser.add_argument("--lambda_li", type=float, default=1.0, help="Weight for late-interaction InfoNCE loss in MetaEmbed-KD")
    parser.add_argument("--lambda_kd", type=float, default=1.0, help="Weight for knowledge distillation loss in MetaEmbed-KD")
    parser.add_argument("--margin", type=float, default=0.2, help="Margin for DeepImpact loss in MetaEmbed-KD")
    parser.add_argument("--kd_temperature", type=float, default=1.0, help="Temperature for InfoNCE and KD in MetaEmbed-KD")
    
    # Negative sampling for MetaEmbed-KD
    parser.add_argument("--num_hard_negatives", type=int, default=8, help="Total number of hard negatives to sample for MetaEmbed-KD (default: 8)")
    parser.add_argument("--top_k_hard", type=int, default=4, help="Number of top hard negatives to always include for MetaEmbed-KD (default: 4)")

    # required for distillation loss with Margin MSE
    parser.add_argument("--qrels_path", type=Path, default=None, help="Path to the qrels file")

    args = parser.parse_args()

    assert not (args.distil_mse and args.distil_kl), "Cannot use both distillation losses at the same time"
    assert not (args.distil_mse and not args.qrels_path), "qrels_path is required for distillation loss with Margin MSE"
    assert not (args.hybrid and args.pairwise), "Cannot use hybrid and pairwise at the same time"
    assert not (args.hybrid and args.cross_encoder), "Cannot use hybrid and cross_encoder at the same time"
    assert not (args.hybrid and args.dense), "Cannot use hybrid and dense at the same time"
    assert not (args.meta_embed and args.hybrid), "Cannot use meta_embed and hybrid at the same time"
    assert not (args.meta_embed and args.pairwise), "Cannot use meta_embed and pairwise at the same time"
    assert not (args.meta_embed and args.cross_encoder), "Cannot use meta_embed and cross_encoder at the same time"
    assert not (args.meta_embed and args.dense), "Cannot use meta_embed and dense at the same time"
    assert not (args.meta_embed and args.distil_mse), "Cannot use meta_embed and distil_mse at the same time"
    assert not (args.meta_embed and args.distil_kl), "Cannot use meta_embed and distil_kl at the same time"

    # pass all argparse arguments to run() as kwargs
    run(**vars(args))
