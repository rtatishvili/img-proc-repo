import matplotlib.pyplot as plt


def plot_multiple_arrays(im_list, figure_title, labels):
    """
    plots a 2D list of images into a matrix shape.

    :param im_list: 2D list, in the shape of matrix (list of (row) list of images)
    :param figure_title: Title for the figure
    :param labels: labels in the sequence of the image; left to right, top to bottom
    """
    figure = plt.figure()
    figure.suptitle(figure_title, fontweight='bold', fontsize=18)
    figure.canvas.set_window_title(figure_title)
    max_row_size = max([len(x) for x in im_list])
    max_col_size = len(im_list)  # TODO: Gigiti

    subplot = 1

    for row_index, each_row in enumerate(im_list):
        for col_index, each in enumerate(each_row):
            # print row_index
            # print '*'
            # print max_col_size
            # print '+'
            # print col_index
            # print '-----'
            a = figure.add_subplot(max_col_size, max_row_size, subplot)
            a.set_title(labels[row_index * max_row_size + col_index], fontsize=10)
            a.imshow(each, cmap=plt.get_cmap('gray'))
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            subplot += 1

    plt.show()