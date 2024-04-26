import pytorch_lightning as pl


class PrintAccuracyAndLossCallback(pl.Callback):
    def on_epoch_end(self, trainer, pl_module):
        # 获取当前 epoch 的训练损失值
        train_loss = trainer.callback_metrics['train_loss']

        # 获取当前 epoch 的准确率
        val_accuracy = trainer.callback_metrics['val_accuracy']

        # 打印当前 epoch 的准确率和损失值
        print(f"Epoch {trainer.current_epoch}: Train Loss {train_loss:.4f}, Val Accuracy {val_accuracy:.4f}")
