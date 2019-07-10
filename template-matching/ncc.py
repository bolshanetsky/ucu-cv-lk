import cv2
import numpy as np
import glob

def template_matching_ncc(frame, template):
    '''
    NCC template matching algorith
    '''
    heigth, width = frame.shape
    heigth_t, width_t = template.shape
    
    score = np.empty((heigth-heigth_t, width-width_t))

    frame = np.array(frame, dtype="float")
    template = np.array(template, dtype="float")

    for dy in range(0, heigth - heigth_t):
        for dx in range(0, width - width_t):
            roi = frame[dy:dy + heigth_t, dx:dx + width_t]
            num = np.sum(roi * template)
            den = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(template ** 2)) 
            if den == 0: score[dy, dx] = 0
            score[dy, dx] = num / den

    point = np.unravel_index(score.argmax(), score.shape)

    return (point[1], point[0])

def load_images(path):
    '''
    Loads images into array
    '''
    image_files = sorted(glob.glob(path))
    images = []
    for idx, file in enumerate(image_files):
        images.append(cv2.imread(file, cv2.IMREAD_COLOR))
    return images


def main():
    TEST_DATA_PATH = 'images/'

    # TEST
    test_images = load_images(TEST_DATA_PATH + "*.jpg")
    template = cv2.imread(TEST_DATA_PATH + "/template.png", cv2.IMREAD_GRAYSCALE)

    for frame in test_images:

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)   

        h, w = template.shape

        pt = template_matching_ncc(gray, template)

        cv2.rectangle(frame, (pt[0], pt[1] ), (pt[0] + w, pt[1] + h), (0,0,200), 3)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()