import cv2
import numpy as np
import glob

def template_matching_ssd(frame, template):
    '''
    SSD matching algorithm
    '''
    heigth, width = frame.shape
    heigth_t, width_t = template.shape
   
    score = np.empty((heigth-heigth_t, width-width_t))
  
    for dy in range(0, heigth - heigth_t):
        for dx in range(0, width - width_t):
            diff = (frame[dy:dy + heigth_t, dx:dx + width_t] - template)**2
            score[dy, dx] = diff.sum()

    pt = np.unravel_index(score.argmin(), score.shape)

    return (pt[1], pt[0])

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
    print(len(test_images))
    template = cv2.imread(TEST_DATA_PATH + "/template.png", cv2.IMREAD_GRAYSCALE)
    print(template.shape)

    for frame in test_images:

        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)   

        h, w = template.shape

        pt = template_matching_ssd(gray, template)

        cv2.rectangle(frame, (pt[0], pt[1] ), (pt[0] + w, pt[1] + h), (0,0,200), 3)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()