
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))

##############
## LinearLR ##
##############

from torch.optim.lr_scheduler import LinearLR

LinearLR(optimizer, start_factor=0.01, end_factor=0.1, total_iters=5000) # 5000 is epochs

##############
## CosineLR ##
##############

from torch.optim.lr_scheduler import CosineAnnealingLR

cosine_scheduler = CosineAnnealingLR(optimizer, T_max=5000, eta_min=1e-7) # 5000 is epochs

#######################
## ReduceLROnPlateau ##
#######################

'''
Reduce learning rate when a metric has stopped improving.

torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

patient=5 -> after 5 epochs, if no improvment -> reduce lr
factor=0.1 -> new_lr = lr*0.1
mode='min' ->  lr will be reduced when the quantity monitored (loss) has stopped decreasing

################

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.1)

for epoch in epoch:
    _ = model.train()
    for X_batch, y_batch in train_set:
    #...training...
    
    _ = model.eval()
    #...validating...
    
    scheduler.step(metrics) # Metrics could be avg_val_loss (MUST HAVE METRICS FOR ReduceLROnPlateau)
    
##################
'''

##################
## SequentialLR ##
##################
'''Combine different LR'''

epochs = 10000
warmup_epochs = int(4e-3*epochs) # 0.4% of total epochs
warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=0.1, total_iters=warmup_epochs)

main_epochs = epochs - warmup_epochs
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=main_epochs, eta_min=1e-7)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

####################
## get current LR ##
####################

'''
for epoch in epoch:
    _ = model.train()
    for X_batch, y_batch in train_set:
    #...training...
    
    _ = model.eval()
    #...validating...
    
    scheduler.step(metrics) # Metrics could be avg_val_loss (MUST HAVE METRICS FOR ReduceLROnPlateau)
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current lr: {current_lr}")
'''