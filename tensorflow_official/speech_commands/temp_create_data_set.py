import numpy as np
import random
def verification_utils_prepare_triplet(ground_truth_label, label_count = 12, num_of_triplets = 1000, hard_mode = False, predicted_label = None):
    if hard_mode:
        #
        ind_each_category = [np.nonzero(ll == ground_truth_label) for ll in range(label_count)]
        anchor_category = np.random.randint(label_count, size=num_of_triplets)
        negative_category = [random.choice(list(set(range(label_count) - anchor_this_category))) for anchor_this_category in anchor_category]

        # chose anchor and positive
        anchor_and_positive = [random.choice(ind_each_category[anchor_this_category], 2, replace = False) for anchor_this_category in anchor_category]
        negative = [random.choice(ind_each_category[anchor_this_category], 2, replace = False) for anchor_this_category in negative_category]




