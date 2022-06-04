import numpy as np

def get_page_from_words(word_lists, MAX_IMG_WIDTH=800):
    line_all = []
    line_t = []

    width_t = 0

    for i in word_lists:

        width_t = width_t + i.shape[1] + 16

        if width_t > MAX_IMG_WIDTH:
            line_all.append(np.concatenate(line_t, 1))

            line_t = []

            width_t = i.shape[1] + 16

        line_t.append(i)
        line_t.append(np.ones((i.shape[0], 16)))

    if len(line_all) == 0:
        line_all.append(np.concatenate(line_t, 1))

    max_lin_widths = MAX_IMG_WIDTH  # max([i.shape[1] for i in line_all])
    gap_h = np.ones([16, max_lin_widths])

    page_ = []

    for l in line_all:
        pad_ = np.ones([l.shape[0], max_lin_widths - l.shape[1]])

        page_.append(np.concatenate([l, pad_], 1))
        page_.append(gap_h)

    page = np.concatenate(page_, 0)

    return page * 255
