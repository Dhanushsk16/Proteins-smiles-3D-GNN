import os
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import config
from dataset import MoleculeGraphDataset, CoordinateNoiseTransform, make_splits
from model import PretrainModel

def setup_logger():
    logger = logging.getLogger("pretrain")
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(config.LOG_DIR, "pretrain.log"))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def compute_loss(outputs, data):
    dist_preds, desc_preds = outputs
    
    denoise_loss = F.mse_loss(dist_preds, data.dist_target)
    
    total_loss = denoise_loss
    desc_loss = torch.tensor(0.0, device=denoise_loss.device)
    
    if config.USE_DESCRIPTOR_HEAD and desc_preds is not None:
        if hasattr(data, 'descriptors'):
            desc_loss = F.mse_loss(desc_preds, data.descriptors)
            total_loss = denoise_loss + config.DESCRIPTOR_LOSS_WEIGHT * desc_loss
            
    return total_loss, denoise_loss, desc_loss

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_sq_err = 0.0
    total_edges = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss, _, _ = compute_loss(outputs, data)
        loss.backward()
        optimizer.step()
        
        dist_preds, _ = outputs
        with torch.no_grad():
            batch_sq_err = ((dist_preds - data.dist_target) ** 2).sum().item()
            total_sq_err += batch_sq_err
            total_edges += data.dist_target.shape[0]
        
    train_mse = total_sq_err / total_edges if total_edges > 0 else 0.0
    return train_mse

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_sq_err = 0.0
    total_edges = 0
    
    for data in loader:
        data = data.to(device)
        if not hasattr(data, 'dist_target'): continue
        outputs = model(data)
        
        dist_preds, _ = outputs
        
        batch_sq_err = ((dist_preds - data.dist_target) ** 2).sum().item()
        total_sq_err += batch_sq_err
        total_edges += data.dist_target.shape[0]
        
    avg_mse = total_sq_err / total_edges if total_edges > 0 else 0.0
    avg_rmse = avg_mse ** 0.5
    return avg_mse, avg_rmse

def train():
    logger = setup_logger()
    
    torch.manual_seed(config.RANDOM_SEED)
    
    transform = CoordinateNoiseTransform()
    dataset = MoleculeGraphDataset(transform=transform)
    
    if config.DEBUG_MODE:
        logger.info(f"DEBUG_MODE is True. Taking subset of size {config.SUBSET_SIZE}")
        from torch.utils.data import Subset
        subset_indices = list(range(min(config.SUBSET_SIZE, len(dataset))))
        dataset = Subset(dataset, subset_indices)
        
    train_dataset, val_dataset = make_splits(dataset, config.VAL_SPLIT, config.RANDOM_SEED)
    
    num_workers = config.NUM_WORKERS
    if os.name == 'nt' and num_workers > 0:
        logger.warning("Windows detected. Using num_workers=0 to prevent multiprocessing crashes.")
        num_workers = 0

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    device = torch.device(config.DEVICE)
    model = PretrainModel().to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")
    logger.info(f"Training set: {len(train_dataset)} graphs | Validation set: {len(val_dataset)} graphs")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 30
    
    logger.info(f"Starting training on {device} for {config.NUM_EPOCHS} epochs (early stopping patience={patience_limit})")
    
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_mse = train_one_epoch(model, train_loader, optimizer, device)
        
        val_mse, val_rmse = 0.0, 0.0
        if len(val_dataset) > 0:
            val_mse, val_rmse = evaluate(model, val_loader, device)
            scheduler.step(val_mse)
        else:
            scheduler.step(train_mse)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch:03d} | Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | Val RMSE: {val_rmse:.4f} Å | LR: {current_lr:.2e}")
        
        is_best = False
        if len(val_dataset) > 0 and val_mse < best_val_loss:
            best_val_loss = val_mse
            is_best = True
            patience_counter = 0
        elif len(val_dataset) == 0 and train_mse < best_val_loss:
            best_val_loss = train_mse
            is_best = True
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch % config.CHECKPOINT_EVERY == 0 or epoch == config.NUM_EPOCHS or is_best:
            ckpt_name = f"model_epoch_{epoch}.pt" if not is_best else "model_best.pt"
            enc_name = f"encoder_epoch_{epoch}.pt" if not is_best else "encoder_best.pt"
            
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, ckpt_name))
            torch.save(model.encoder.state_dict(), os.path.join(config.OUTPUT_DIR, enc_name))
            
        if patience_counter >= patience_limit:
            logger.info(f"Early stopping triggered after {epoch} epochs (patience={patience_limit}).")
            break
            
    logger.info("Training complete.")

