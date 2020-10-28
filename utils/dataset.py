class fashionMNISTDataset(Dataset):
    """
    To build a dataset using the fashion MNIST data.
    """
    
    def __init__(self, features, labels, plot=False):
        self.images = features
        self.labels = labels
        self.plot = plot
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx, :]
        label = self.labels[idx]
        sample = (image, label)
        if self.plot:
            plt.imshow(self.images[idx].reshape(28, 28), cmap='gray')
            plt.title(get_label(self.labels[idx]))
        return sample

def get_label(label):
    """
    To get a label as a string when entering a numeric label.
    """
    
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[i] for i in label]