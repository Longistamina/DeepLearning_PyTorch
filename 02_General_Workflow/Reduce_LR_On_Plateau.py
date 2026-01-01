'''
Reduce learning rate when a metric has stopped improving.

torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

patient=5 -> after 5 epochs, if no improvment -> reduce lr
factor=0.1 -> new_lr = lr*0.1
mode='min' ->  lr will be reduced when the quantity monitored (loss) has stopped decreasing

################

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)

for epoch in epoch:
    _ = model.train()
    for X_batch, y_batch in train_set:
    #...training...
    
    _ = model.eval()
    #...validating...
    
    scheduler.step(metrics) # Metrics could be avg_val_loss (MUST HAVE METRICS FOR ReduceLROnPlateau)
'''