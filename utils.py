# input: img --> 2D or 3D array
# output: histogram normalized
def compute_histogram(img):
    if len(img.shape) == 3:
        h, w, d = img.shape
        h_w = h * w
        if d == 3:
            p1 = img[:, :, 0]
            p2 = img[:, :, 1]
            p3 = img[:, :, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    if len(img.shape) == 2:
        h_w, d = img.shape
        if d == 3:
            p1 = img[:, 0]
            p2 = img[:, 1]
            p3 = img[:, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]
    
    # e' corretto 256, non h_w
    histogram = np.zeros(256*d) 
    for i in np.arange(len(planes)):
        p = planes[i]
        for val in np.unique(p):
            count = np.sum(p == val)
            histogram[val + i*256] = count
    histogram = histogram / img.size
    return histogram


# function for Shannon's Entropy    
def entropy(histogram):
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))