import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from torch.utils.data import DataLoader


from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

class LitYolo(LightningModule):
    def __init__(self, batch_size=64):
        super().__init__()
        
        self.lr = config.LEARNING_RATE
        self.weight_decay =config.WEIGHT_DECAY        
        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        self.save_hyperparameters()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss_fn = YoloLoss()


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = self.model(x)
            loss = (
                self.loss_fn(out[0], y0, scaled_anchors[0])
                + self.loss_fn(out[1], y1, scaled_anchors[1])
                + self.loss_fn(out[2], y2, scaled_anchors[2])
            )  
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss

    

    def validation_step(self, batch, batch_idx):
        pass        

    def test_step(self, batch, batch_idx):
        pass
    
    def on_epoch_start(self):
        if self.trainer.training:
            plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, self.scaled_anchors)
            
        
    def on_train_epoch_end(self):
        if config.SAVE_MODEL:
            save_checkpoint(self.model, self.optimizer, filename=config.CHECKPOINT_FILE)                
        epoch = self.trainer.current_epoch
        print(f"Currently epoch {epoch}")
        print("On Train loader:")
        check_class_accuracy(self.model, self.train_dataloader, threshold=config.CONF_THRESHOLD)

       
        
    def on_validation_epoch_end(self):      
        
        epoch = self.trainer.current_epoch
        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(self.model, self.test_dataloader(), threshold=config.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_dataloader(),
                self.model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")    
        
        

   
    def lr_finder(self, optimizer, num_iter=50):
        from torch_lr_finder import LRFinder
        
        
        self.scaled_anchors = (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        ).to(config.DEVICE)

        def criterion(out, y):
            y0, y1, y2 = (
                    y[0].to(config.DEVICE),
                    y[1].to(config.DEVICE),
                    y[2].to(config.DEVICE),
                )
            loss = (
                        self.loss_fn(out[0], y0, scaled_anchors[0])
                        + self.loss_fn(out[1], y1, scaled_anchors[1])
                        + self.loss_fn(out[2], y2, scaled_anchors[2])
                    )
            return loss
        
        lr_finder = LRFinder(self.model, self.optimizer, criterion, device=config.DEVICE)
        lr_finder.range_test(self.train_dataloader(), end_lr=1, num_iter=num_iter, step_mode="exp")
        ax, suggested_lr = lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset() # to reset the model and optimizer to their initial state
        return suggested_lr
    
    def configure_optimizers(self):
        
        
        suggested_lr = self.lr_finder(self.optimizer) #check on self.train_dataloader
        
        steps_per_epoch = len(self.train_dataloader())
        scheduler_dict = {
            "scheduler":  OneCycleLR(
        optimizer, max_lr=suggested_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=self.trainer.max_epochs, 
        pct_start=5/self.trainer.max_epochs,
        three_phase=False,
        div_factor=100,
        final_div_factor=100,
        anneal_strategy='linear',
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
    
    
    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        
        # download       
        from dataset import YOLODataset
        IMAGE_SIZE = config.IMAGE_SIZE
        train_csv_path=config.DATASET + "/train.csv"
        test_csv_path=config.DATASET + "/test.csv"
        
        self.train_dataset = YOLODataset(
            train_csv_path,
            transform=config.train_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
        )
        
        self.test_dataset = YOLODataset(
        test_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        )
        
        self.val_dataset = YOLODataset(
        train_csv_path,
        transform=config.test_transforms,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        anchors=config.ANCHORS,
        )
        
        if config.LOAD_MODEL:
            load_checkpoint(
                config.CHECKPOINT_FILE, self.model, self.optimizer, config.LEARNING_RATE)            
        

    def setup(self, stage=None):
        pass 
       

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            shuffle=True,
            drop_last=False,
        )   

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )
    

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=config.BATCH_SIZE,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            persistent_workers=True,
            shuffle=False,
            drop_last=False,
        )