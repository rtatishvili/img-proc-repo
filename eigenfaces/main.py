import src.image_loader as iml
import src.training as tr

if __name__ == '__main__':
    images = iml.load_images(0, 200)
    (X_train, X_test) = tr.divide_dataset_into_train_test(images)

    print X_train.shape
    print X_test.shape
