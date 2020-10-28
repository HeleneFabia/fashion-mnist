def test_model(net, test_ds, device):
    """
    Testing a model with the test set
    """
    net = net.to(device)
    net.eval()
    
    #test_predictions = torch.empty(0,10).cuda()   
    correct = 0
    best_val_acc = 0
    loss_examples = []
    idx_false_preds = []

    net.eval()
    with torch.no_grad():
        X_test = test_ds.images.to(device)
        y_test = test_ds.labels.to(device)
        test_preds = net(X_test)
        test_loss = loss(test_preds, y_test).detach().item()

        predicted = torch.max(test_preds.data, 1)[1]
        correct += (predicted == y_test).sum()
            
        correct = correct.float()      
        correct_epoch = (correct/(10000))
        
        for i in range(len(predicted)):
            if predicted[i] != y_test[i]:
                idx_false_preds.append(i)
                
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {correct_epoch:.4f}')
    
    return best_val_acc, test_preds, idx_false_preds


def get_results(batch_size, test_ds_cnn, verbose=False):
    """
    Get predictions, true labels and visualization of images of a random batch of the test set.
    """

    x, y = get_test_batch(batch_size=25, test_ds_cnn=test_ds_cnn)
    
    with torch.no_grad():
        pred = model_testing_cnn(x)
    pred = pred.cpu().detach().numpy()
    
    predictions = []
    probabilities = []
    
    for i, pred in enumerate(pred):
        pred_soft = softmax(pred)
        pred_label = np.argmax(pred_soft)
        pred_prob = pred_soft[pred_label]
        predictions.append(pred_label)
        probabilities.append(pred_prob)
        
        if verbose:
            print('Predicted Label:', pred_label)
            print('Predicted Probability:', pred_soft[pred_label])
            print('Actual Label:', y[i])
            print('___')
    
    show_predicted_images(x.reshape(batch_size,28,28), int(batch_size/5), 5, true=(get_label)(y), pred=get_label(predictions), probabilities=probabilities)
    

def get_test_batch(batch_size, test_ds_cnn):
    """
    Get a random batch of size batch_size.
    """

    assert (batch_size%5 == 0),"Choose batch_size that is multiple of 5."
    test_dl = data.DataLoader(test_ds_cnn, batch_size=batch_size, shuffle=True)
    for batch in test_dl:
        x = batch[0].cuda()
        y = batch[1].detach().cpu().numpy()
        break
    return x, y


def show_predicted_images(images, num_rows, num_cols, true=None, pred=None, probabilities=None, scale=1.5):
    """
    Show image alongside the predicted and true label.
    """
    figsize = (num_cols * 2, num_rows * 1.5)
    figure, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    figure.tight_layout()
    for i, (ax, images) in enumerate(zip(axes, images.cpu())):
        ax.imshow(np.array(images), cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if true and pred:
            ax.set_title(f'Label: {true[i]}\nPred: {pred[i]} ({probabilities[i]:.2f})')
    plt.tight_layout()
    return axes


def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()