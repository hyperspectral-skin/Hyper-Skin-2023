import torch 
import time 
import gc

class create(object):
    def __init__(self, model_architecture, device,
                 loss_fn, optimizer, scheduler, epochs,
                 logger, model_saved_path):
        self.model = model_architecture
        self.device = device

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.epochs = epochs
        self.logger = logger
        self.model_saved_path = model_saved_path


    def train_fn(self, current_epoch, train_loader):
        self.model.train() 
        total_loss = 0.0 

        for batch, data in enumerate(train_loader):  
            x, y = data 
            x, y = x.float().to(self.device), y.float().to(self.device)

            # forward pass and compute the training loss
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            # zero the gradients accumulated from the previous operation
            # backward pass, and update the model parameter
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step() 
            self.scheduler.step()

            total_loss += loss.item() 
            print(f"\r[Training Epoch {current_epoch+1} - Batch {batch}] >> loss: {float(total_loss/(batch+1)):6f} ", end="")
                    
        return total_loss / len(train_loader)

    def valid_fn(self, valid_loader):
        self.model.eval() 
        total_loss = 0.0 

        for batch, data in enumerate(valid_loader):
            x, y = data 
            x, y = x.float().to(self.device), y.float().to(self.device)

            with torch.no_grad():
                pred = self.model(x)
            loss = self.loss_fn(pred, y)
            total_loss += loss.item() 
            
        return total_loss / len(valid_loader)

    def train(self, train_loader, valid_loader, best_valid_loss):
        self.model.to(self.device)
        history = {
            "train_loss": [],
            "val_loss": []
        }
        start_time = time.time() 

        for epoch in range(self.epochs):
            train_loss = self.train_fn(epoch, train_loader)
            valid_loss = self.valid_fn(valid_loader)
            print(f"\r[Training Epoch {epoch+1}/{self.epochs}] >> loss: {train_loss:6f}", end="")
            print(f'  (validation loss: {valid_loss:.6f})')
            
            if valid_loss < best_valid_loss:
                self.save_model()
                best_valid_loss = valid_loss 

            history['train_loss'].append(train_loss)
            history['val_loss'].append(valid_loss)

            log_history = {
                'train-loss': train_loss,
                'valid_loss': valid_loss,
            }
            self.logger.info(f"Epoch [{epoch}/{self.epochs}]: {log_history}")


        gc.collect()
        torch.cuda.empty_cache()
        self.model.cpu()
        log_msg = f"Total training time: {time.time() - start_time}\n" + \
                   "==========================================================================\n"
        print(log_msg)
        self.logger.info(log_msg)

        return best_valid_loss, history

    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_saved_path)
        print(f"Save model ...")


    def load_model(self, model_template):
        checkpoint = torch.load(self.model_saved_path)
        self.model = model_template
        self.model.load_state_dict(checkpoint)

        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()

    def evaluate(self, test_loader, eval_fn):
        self.model.to(self.device)
        self.model.eval() 
        predictions = []
        results = {
            "mae": [],
            "sam": []
        }
        for batch, data in enumerate(test_loader):
            x, y = data 
            x, y = x.float().to(self.device), y.float().to(self.device)

            # don't need to calculate the gradients during evaluation
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred.cpu().detach().numpy())
            loss = self.loss_fn(pred, y)
            sam = eval_fn(pred, y)

            results["mae"].append(loss.item())
            results["sam"].append(sam.item())

            log_results = {
                'mae': loss.item(),
                'sam': sam.item(),
            }
            self.logger.info(f"Evaluation [{batch}/{len(test_loader)}]: {log_results}")
            print(f"Evaluation [{batch}/{len(test_loader)}]: {log_results}")
        self.model.cpu()
        del x, y, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        return results, predictions