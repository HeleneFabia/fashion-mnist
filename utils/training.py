def train_model(net, train_dl, val_dl, num_epochs, learning_rate, device, save_model=False, early_stopping_patience=8):
    
    net = net.to(device)
    
    train_epoch_loss, valid_epoch_loss, acc_epoch = [], [], []
    best_val_acc = 0
    
    optimizer = torch.optim.Adam(net.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5, verbose=True)
    early_stopping_count = 0
        
    for epoch in range(num_epochs):  
        
        net.train()
        correct = 0
    
        train_batch_loss, valid_batch_loss = [], []
        
        for batch in train_dl:
            
            X_train = batch[0].to(device)
            y_train = batch[1].to(device)
            optimizer.zero_grad()
            preds = net(X_train)
            l = loss(preds, y_train)
            l.backward()
            optimizer.step()
            
            train_batch_loss.append(l.detach().item())
        
        train_epoch_loss.append(np.mean(train_batch_loss))
        
        net.eval()
        with torch.no_grad():
            for batch in val_dl:
                X_val = batch[0].to(device)
                y_val = batch[1].to(device)
                val_preds = net(X_val)
                #if epoch == (num_epochs-1):
                    #val_predictions = torch.cat((val_predictions, val_preds), 0)
                valid_batch_loss.append(loss(val_preds, y_val).detach().item())

                predicted = torch.max(val_preds.data, 1)[1]
                correct += (predicted == y_val).sum()
            correct = correct.float()      
            correct_epoch = (correct/len(val_dl.dataset))
            acc_epoch.append(correct_epoch)           
            
        scheduler.step(correct_epoch)
        
        valid_epoch_loss.append(np.mean(valid_batch_loss))
        
        if save_model==True:
            if correct_epoch >= best_val_acc:
                #print(f'Validation accuracy has improved from {best_val_acc:.4f} to {correct_epoch:.4f}. Saving model...')
                torch.save(net.state_dict(), 'best_model.pt')
            
        if correct_epoch >= best_val_acc:
            best_val_acc = correct_epoch
            early_stopping_count = 0
        else:
            early_stopping_count += 1
            
        print(f'Ep: {epoch+1}/{num_epochs} | Train Loss: {train_epoch_loss[-1]:.4f} | Val Loss: {valid_epoch_loss[-1]:.4f} | Val Acc: {correct_epoch:.4f}')
        
        if early_stopping_count >= early_stopping_patience:
            print("Early Stopping")
            break
            
    print(f'Best Validation Accuracy: {best_val_acc:.4f}')
    plot_learning_curve((epoch + 1), train_epoch_loss, valid_epoch_loss, acc_epoch, 'Loss/Accuracy');

    return train_epoch_loss, valid_epoch_loss, acc_epoch

def plot_learning_curve(num_epochs, train_loss, valid_loss, acc, ylabel):
    plt.plot(range(num_epochs), train_loss, label='train loss')
    plt.plot(range(num_epochs), valid_loss, label='valid loss')
    plt.plot(range(num_epochs), acc, label='valid accuracy')
    plt.xticks(range(num_epochs), range(1, num_epochs + 1))
    plt.ylabel(ylabel)
    plt.xlabel('Epochs')
    plt.legend()