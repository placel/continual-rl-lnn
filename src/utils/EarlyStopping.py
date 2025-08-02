class EarlyStopping:

    def __init__(self, patience=10, min_delta=0.0, mode='max', verbose=False, restore_weights_flag=True):
        self.patience = patience
        self.min_delta = min_delta 
        self.mode = mode # Min or Max
        self.verbose = verbose
        self.best_score = None
        self.wait = 0
        self.stop = False
        self.best_weights = None # Used for restoring_weights
        self.restore_weights_flag = restore_weights_flag

    def update(self, score, model):
        
        # If the score is 0.0 the model isn't learning, and early_stopping should not be considered
        if score == 0.0:
            return self.stop
        
        improved = False
        # On first run, initialize the best_score and store model weights in best_weights
        if self.best_score is None:
            self.best_score = score
            improved = True
            if model is not None:
                self.best_weights = self.get_state_dict(model)

        # If mode == 'max' we are trying to maximize the value (i.e. average_reward)
        # Otherwise mode =='min' means we are trying to minimize (i.e. loss)
        else:
            if self.mode == 'max':
                if score > self.best_score + self.min_delta:
                    improved = True
            else:
                if score < self.best_score - self.min_delta:
                    improved = True

        if improved:
            self.best_score = score

            if self.restore_weights_flag:
                self.best_weights = self.get_state_dict(model) # Store the best models weights
            
            self.wait = 0 # Reset the wait counter

            if self.verbose:
                print(f'EarlyStopping: new best score: {self.best_score}.')
        else:
            self.wait += 1 # Increment the wait counter
            
            if self.verbose:
                print(f'EarlyStopping wait increment: {self.wait}')
            
            if self.wait >= self.patience:
                self.stop = True
        
        # Return a bool of if the model has triggered EarlyStopping
        return self.stop

    def get_state_dict(self, model):
        # Iterate over every tensor stored in the model.state_dict(), send it to cpu() and clone it (prevents device mistmatch errors)
        return {k: v.cpu().clone() for k, v in model.state_dict().items()}

    def restore_weights(self, model):

        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
        else:
            print("No weights to restore.")