import random

import numpy as np

def get_page_from_words(word_lists,
                        max_line_width=800,
                        horizontal_space=16,
                        vertical_space=16):
    """ line height, line width

    - mcmc - choose offset from previous line
           - sum offsets
           - find max/min
           - this is the line thickness
    Args:
        word_lists:
        max_line_width:
        horizontal_space:

    Returns:

    """
    page_lines = []
    current_line = []
    y_prev_line_total_height = 0
    ys = []
    xs = []
    localization = []
    x_start = 0

    """
    For Y's:
        # line starting position -- can't be known until all offsets are calculated
        # where the previous line left off
        # 
    """
    for ii,word in enumerate(word_lists):
        xs.append(x_start)
        x_end = x_start + word.shape[1]
        offset = random.randint(-1,1)
        ys.append(offset)

        # START NEW LINE
        if x_end > max_line_width:
            ys_sum = np.cumsum(ys)
            ys_sum -= np.min(ys_sum)
            y_height_for_current_line = word.shape[0] + max(ys_sum)
            line_tensor = np.ones([y_height_for_current_line, x_end], dtype=np.int64)
            for i,_word in enumerate(current_line):
                line_tensor[y_height_for_current_line+ys_sum[i]:_word.shape[0], xs[i]:_word.shape[1]] = _word
            #page_lines.append(np.concatenate(current_line, 1))
            current_line = []
            x_end = word.shape[1] + horizontal_space
            for l in localization[-len(ys):]:
                l["ll"][1] += y_height_for_current_line

            ys = []
            xs = 0
            x_start=0
            # TODO: Variable vertical space
            y_prev_line_total_height += y_height_for_current_line + vertical_space

        current_line.append(word)
        current_line.append(np.ones((word.shape[0], horizontal_space)))

        # Once the line is done, we need to go back and add the height of the total line back in
        localization.append({"ll": [x_start, offset + y_prev_line_total_height], "ur": (x_end, y_prev_line_total_height), "line_number": len(page_lines), "line_word_idx":ii})
        x_start = x_end + horizontal_space

    if len(page_lines) == 0:
        page_lines.append(np.concatenate(current_line, 1))

    # max_line_width = max([i.shape[1] for i in line_all])
    vertical_line_space = np.ones([vertical_space, max_line_width])
    page_ = []

    #
    for line in page_lines:
        right_pad_ = np.ones([line.shape[0], max_line_width - line.shape[1]])

        page_.append(np.concatenate([line, right_pad_], 1))
        page_.append(vertical_line_space)

    page = np.concatenate(page_, 0)

    return page * 255
