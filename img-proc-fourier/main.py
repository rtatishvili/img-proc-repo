from fourier.image_io import read_image


def task_1_1():
    image = read_image('test/bauckhage.jpg', False)
    image.show()

if __name__ == '__main__':
    task_1_1()
    